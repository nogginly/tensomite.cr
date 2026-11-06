require "json"
require "file_utils"
require "./corpus"

module Finetune
  # How a run trains: the model it starts from, the text, and every setting that
  # decides its steps. Saved in the run directory, so a resumed run uses the same.
  class Settings
    include JSON::Serializable

    # The pretrained GPT-2's directory, as `scripts/fetch_gpt2.sh` downloads it.
    property model : String = "models/gpt2"
    # The model's attention heads; unless given, `n_head` in its `config.json`,
    # or 12 without one.
    property heads : Int32? = nil
    # The text to fine-tune on.
    property data : String = "data/shakespeare/input.txt"
    property batch : Int32 = 8
    property context : Int32 = 256
    property steps : Int32 = 1000
    # The peak learning rate, reached after `warmup` steps, then decayed along a
    # cosine to `min_lr` at the last step.
    property lr : Float64 = 1e-4
    property min_lr : Float64 = 1e-5
    property warmup : Int32 = 50
    property weight_decay : Float64 = 0.01
    # The share of activations dropout zeroes in training; GPT-2 was trained
    # with 0.1. 0 for a run whose settings predate it.
    property dropout : Float64 = 0.0
    property beta1 : Float64 = 0.9
    property beta2 : Float64 = 0.999
    property seed : Int32 = 1337
    # Steps between validations, each over `eval_batches` fixed batches.
    property eval_every : Int32 = 100
    property eval_batches : Int32 = 8
    # Validations without a new best before stopping; 0 never stops early.
    property patience : Int32 = 0
    # What each validation's sample continues, and how many tokens it writes.
    property sample_prompt : String = "\n"
    property sample_length : Int32 = 40

    def initialize
    end

    # Returns the learning rate of step *step*, counted from 0: a linear warm-up
    # to `lr`, then a cosine decay to `min_lr` at the last step.
    def lr_at(step : Int32) : Float64
      return lr * (step + 1) / warmup if step < warmup
      progress = (step - warmup) / Math.max(1, steps - 1 - warmup)
      min_lr + 0.5 * (lr - min_lr) * (1 + Math.cos(Math::PI * Math.min(1.0, progress)))
    end
  end

  # What a checkpoint records beside the weights and the optimizer's state.
  record Progress, step : Int32, best : Float64?, since_best : Int32 do
    include JSON::Serializable
  end

  # Fine-tunes a pretrained GPT-2, in a run directory that holds everything the
  # run needs to resume:
  #
  # - `settings.json`, the run's `Settings`;
  # - `tokens.bin` and `split.json`, the tokenised text and where validation
  #   begins, written once;
  # - `checkpoint/`, the weights, AdamW's state and the `Progress`, replaced
  #   whole at each checkpoint;
  # - `best/` and `model/`, the weights at the best validation loss and at the
  #   end, each with the tokeniser and config, so `apps/server` serves them.
  #
  # Each step's batch depends only on the seed and the step, so a run resumed
  # from a checkpoint takes the same steps, with the same bits on one device, as
  # one that never stopped.
  class Trainer
    getter settings : Settings
    getter model : Tensomite::GPT2
    getter optimizer : Tensomite::AdamW
    getter tokenizer : Tensomite::BPE
    getter progress : Progress

    @split : Int64
    @corpus : Corpus

    # Returns a trainer for *run*, resuming from its checkpoint if it has one.
    # A new run takes *settings* and tokenises the text; a resumed one reads its
    # own settings, and *settings* are ignored.
    def initialize(@run : String, settings : Settings, @device : Tensomite::Device = Tensomite::CPU.default)
      Dir.mkdir_p(@run)
      settings_path = File.join(@run, "settings.json")
      @settings = File.exists?(settings_path) ? Settings.from_json(File.read(settings_path)) : settings
      File.write(settings_path, @settings.to_pretty_json) unless File.exists?(settings_path)
      @tokenizer = Tensomite::BPE.from_files(File.join(@settings.model, "vocab.json"), File.join(@settings.model, "merges.txt"))
      @split = prepare_corpus
      @corpus = Corpus.new(File.join(@run, "tokens.bin"))
      checkpoint = File.join(@run, "checkpoint")
      resuming = File.exists?(File.join(checkpoint, "progress.json"))
      @model = load_model(resuming ? File.join(checkpoint, "model.safetensors") : File.join(@settings.model, "model.safetensors"))
      @optimizer = Tensomite::AdamW.new(@model.params, lr: @settings.lr, beta1: @settings.beta1, beta2: @settings.beta2,
        weight_decay: @settings.weight_decay)
      @progress = Progress.new(0, nil, 0)
      if resuming
        @progress = Progress.from_json(File.read(File.join(checkpoint, "progress.json")))
        @optimizer.restore(Tensomite::SafeTensors.open(File.join(checkpoint, "optimizer.safetensors")), steps: @progress.step)
      end
    end

    # Returns whether the run has taken all its steps.
    def done? : Bool
      progress.step >= settings.steps
    end

    # Returns the inputs `[batch, context]` and next-token targets of step
    # *step*: windows of the training text drawn by a generator seeded with the
    # run's seed and the step, so any step's batch can be made again.
    def batch(step : Int32) : {Tensomite::Tensor, Tensomite::Tensor}
      random = Random::PCG32.new(settings.seed.to_u64, step.to_u64)
      span = @split - settings.context - 1
      raise Tensomite::Error.new("the training text is shorter than one window") if span <= 0
      windows = Array.new(settings.batch) { @corpus.window(random.rand(span), settings.context + 1) }
      inputs = windows.flat_map(&.[0, settings.context])
      targets = windows.flat_map(&.[1, settings.context])
      {Tensomite::Tensor.from(inputs, [settings.batch, settings.context], @device),
       Tensomite::Tensor.from(targets, [settings.batch * settings.context], @device)}
    end

    # Takes the next step and returns its loss.
    def step : Float32
      inputs, targets = batch(progress.step)
      optimizer.lr = settings.lr_at(progress.step)
      loss = optimizer.step(model.call(inputs, dropout: dropout(progress.step)).cross_entropy(targets))
      @progress = @progress.copy_with(step: progress.step + 1)
      GC.collect
      loss
    end

    # Returns the dropout of step *step*, seeded by the run's seed and the step,
    # or `nil` when the run has none.
    private def dropout(step : Int32) : Tensomite::Dropout?
      return if settings.dropout == 0
      Tensomite::Dropout.new(settings.dropout, seed: Tensomite::Dropout.mix((settings.seed.to_u64 << 32) | step.to_u64))
    end

    # Returns the mean loss over the validation text's `eval_batches` fixed
    # batches, evenly spaced through it. Collects after each batch, as `#step`
    # does, so that one batch's device memory is free before the next.
    def validation_loss : Float64
      span = @corpus.size - @split - settings.context - 1
      raise Tensomite::Error.new("the validation text is shorter than one window") if span <= 0
      count = settings.eval_batches * settings.batch
      total = (0...settings.eval_batches).sum do |b|
        windows = Array.new(settings.batch) { |i| @corpus.window(@split + span * (b * settings.batch + i) // count, settings.context + 1) }
        inputs = Tensomite::Tensor.from(windows.flat_map(&.[0, settings.context]), [settings.batch, settings.context], @device)
        targets = Tensomite::Tensor.from(windows.flat_map(&.[1, settings.context]), [settings.batch * settings.context], @device)
        loss = model.call(inputs).cross_entropy(targets).to_a(Float32)[0].to_f64
        GC.collect
        loss
      end
      total / settings.eval_batches
    end

    # Records the validation *loss*, saving the weights as `best/` when it is the
    # lowest yet, and returns whether the run should stop early.
    def record(loss : Float64) : Bool
      if (best = progress.best).nil? || loss < best
        @progress = progress.copy_with(best: loss, since_best: 0)
        export(File.join(@run, "best"))
        false
      else
        @progress = progress.copy_with(since_best: progress.since_best + 1)
        settings.patience > 0 && progress.since_best >= settings.patience
      end
    end

    # Returns a sample continuing the settings' prompt, from the model as it is.
    def sample : String
      prompt = tokenizer.encode(settings.sample_prompt)
      ids = model.generate(prompt, settings.sample_length, Tensomite::Sampler.new(temperature: 0.8, top_k: 40, seed: settings.seed))
      GC.collect
      tokenizer.decode(ids)
    end

    # Saves the weights, AdamW's state and the progress as `checkpoint/`,
    # writing them to a new directory first and then replacing the old one, so
    # an interruption leaves one whole checkpoint or the other.
    def checkpoint : Nil
      fresh = File.join(@run, "checkpoint.new")
      FileUtils.rm_rf(fresh)
      Dir.mkdir_p(fresh)
      model.save(File.join(fresh, "model.safetensors"))
      Tensomite::SafeTensors.write(File.join(fresh, "optimizer.safetensors"), optimizer.state)
      File.write(File.join(fresh, "progress.json"), progress.to_json)
      old = File.join(@run, "checkpoint")
      stale = File.join(@run, "checkpoint.old")
      FileUtils.rm_rf(stale)
      File.rename(old, stale) if Dir.exists?(old)
      File.rename(fresh, old)
      FileUtils.rm_rf(stale)
    end

    # Writes the weights to *dir* with the pretrained model's tokeniser files and
    # config, as a directory `apps/server` serves.
    def export(dir : String) : Nil
      Dir.mkdir_p(dir)
      model.save(File.join(dir, "model.safetensors"))
      %w[vocab.json merges.txt config.json].each do |name|
        source = File.join(settings.model, name)
        FileUtils.cp(source, File.join(dir, name)) if File.exists?(source)
      end
    end

    # Tokenises the text into `tokens.bin` the first time, and returns where its
    # last tenth, the validation text, begins.
    private def prepare_corpus : Int64
      tokens, split = File.join(@run, "tokens.bin"), File.join(@run, "split.json")
      unless File.exists?(split)
        ids = tokenizer.encode(File.read(settings.data))
        Corpus.write(tokens, ids)
        File.write(split, {validation_starts: ids.size.to_i64 * 9 // 10}.to_json)
      end
      JSON.parse(File.read(split))["validation_starts"].as_i64
    end

    private def load_model(path : String) : Tensomite::GPT2
      config = File.join(settings.model, "config.json")
      heads = settings.heads || (File.exists?(config) ? JSON.parse(File.read(config))["n_head"].as_i : 12)
      Tensomite::GPT2.load(path, heads: heads, device: @device)
    end
  end
end
