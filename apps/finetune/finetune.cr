# Fine-tunes a pretrained GPT-2 on a text file, in a run directory it can resume
# from, and exports the result for `apps/server` to serve.
#
#     ops br finetune                     # builds bin/release/finetune
#     bin/release/finetune --out runs/shakespeare --data data/shakespeare/input.txt --device metal
#     bin/release/finetune --out runs/shakespeare --device metal          # resumes
#     bin/release/server --model runs/shakespeare/best
#
# A new run takes its settings from the options; a resumed one reads them from
# its directory, and only `--device` and `--checkpoint-every` apply. See
# `apps/finetune/README.md`.
require "option_parser"
require "./src/trainer"
require "../../src/tensomite/blas"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../../src/tensomite/metal"
{% end %}
{% if flag?(:cuda) %}
  require "../../src/tensomite/cuda"
{% end %}

devices = {"cpu" => -> { Tensomite::CPU.default.as(Tensomite::Device) }, "blas" => -> { Tensomite::BLAS.default.as(Tensomite::Device) }}
{% if flag?(:darwin) && flag?(:aarch64) %}
  devices["metal"] = -> { Tensomite::Metal.default.as(Tensomite::Device) }
{% end %}
{% if flag?(:cuda) %}
  devices["cuda"] = -> { Tensomite::CUDA.default.as(Tensomite::Device) }
{% end %}

settings = Finetune::Settings.new
run, device_name, checkpoint_every = nil.as(String?), "blas", 100
OptionParser.parse do |parser|
  parser.banner = "Usage: finetune --out RUN [options]"
  parser.on("--out DIR", "The run's directory, created or resumed") { |dir| run = dir }
  parser.on("--device NAME", "#{devices.keys.join(", ")} (#{device_name})") { |name| device_name = name }
  parser.on("--checkpoint-every N", "Steps between checkpoints (#{checkpoint_every})") { |value| checkpoint_every = value.to_i }
  parser.separator "\nFor a new run:"
  parser.on("--model DIR", "Pretrained GPT-2 (#{settings.model})") { |dir| settings.model = dir }
  parser.on("--heads N", "Attention heads (n_head in the model's config.json)") { |value| settings.heads = value.to_i }
  parser.on("--data PATH", "Text to fine-tune on (#{settings.data})") { |path| settings.data = path }
  parser.on("--batch N", "Sequences per step (#{settings.batch})") { |value| settings.batch = value.to_i }
  parser.on("--context N", "Tokens per sequence (#{settings.context})") { |value| settings.context = value.to_i }
  parser.on("--steps N", "Training steps (#{settings.steps})") { |value| settings.steps = value.to_i }
  parser.on("--lr X", "Peak learning rate (#{settings.lr})") { |value| settings.lr = value.to_f }
  parser.on("--min-lr X", "Final learning rate (#{settings.min_lr})") { |value| settings.min_lr = value.to_f }
  parser.on("--warmup N", "Warm-up steps (#{settings.warmup})") { |value| settings.warmup = value.to_i }
  parser.on("--weight-decay X", "AdamW's weight decay (#{settings.weight_decay})") { |value| settings.weight_decay = value.to_f }
  parser.on("--seed N", "Random seed (#{settings.seed})") { |value| settings.seed = value.to_i }
  parser.on("--eval-every N", "Steps between validations (#{settings.eval_every})") { |value| settings.eval_every = value.to_i }
  parser.on("--eval-batches N", "Batches per validation (#{settings.eval_batches})") { |value| settings.eval_batches = value.to_i }
  parser.on("--patience N", "Validations without a new best before stopping, 0 for never (#{settings.patience})") { |value| settings.patience = value.to_i }
  parser.on("--prompt TEXT", "What each validation's sample continues") { |text| settings.sample_prompt = text }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
  parser.invalid_option { |flag| abort "unknown option #{flag}\n#{parser}" }
end
dir = run
abort "finetune needs --out DIR" unless dir
device = devices[device_name]?.try(&.call) || abort "unknown device #{device_name}; this build has #{devices.keys.join(", ")}"

Dir.mkdir_p(dir)
log = File.open(File.join(dir, "log.txt"), "a")
say = ->(line : String) { puts line; log.puts line; log.flush }
trainer = Finetune::Trainer.new(dir, settings, device)
s = trainer.settings
start = trainer.progress.step
say.call(start == 0 ? "new run in #{dir} on #{device_name}" : "resuming #{dir} at step #{start} on #{device_name}")
say.call("#{s.model} on #{s.data}: batch #{s.batch} × #{s.context}, #{s.steps} steps, lr #{s.lr} to #{s.min_lr}")

since = Time.instant
until trainer.done?
  loss = trainer.step
  step = trainer.progress.step
  if step % 10 == 0 || trainer.done?
    ms = (Time.instant - since).total_milliseconds / 10
    say.call("step #{step}: loss #{loss.round(4)}, lr #{s.lr_at(step - 1).round(7)}, #{ms.round(0)} ms/step")
    since = Time.instant
  end
  stop = false
  if step % s.eval_every == 0 || trainer.done?
    validation = trainer.validation_loss
    stop = trainer.record(validation)
    best = trainer.progress.best.try(&.round(4))
    say.call("step #{step}: validation loss #{validation.round(4)}, best #{best}")
    say.call("step #{step}: sample #{trainer.sample.inspect}")
  end
  trainer.checkpoint if step % checkpoint_every == 0 || trainer.done? || stop
  if stop
    say.call("step #{step}: no new best in #{s.patience} validations; stopping")
    break
  end
end
trainer.export(File.join(dir, "model"))
say.call("done: #{dir}/best holds the best weights, #{dir}/model the last; serve one with apps/server")
