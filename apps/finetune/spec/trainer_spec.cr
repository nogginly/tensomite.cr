require "spec"
require "file_utils"
require "../src/trainer"

include Tensomite

FINETUNE_FIXTURE = File.expand_path("#{__DIR__}/../../../spec/fixtures/tiny_gpt2")

# A model directory with the tiny GPT-2's files, and a text to train on.
private def workspace : String
  dir = File.tempname("finetune")
  Dir.mkdir_p(dir)
  %w[model.safetensors vocab.json merges.txt].each { |name| FileUtils.cp(File.join(FINETUNE_FIXTURE, name), File.join(dir, name)) }
  words = %w[the quick brown fox jumps over a lazy dog and then it runs far away into hills]
  random = Random.new(5)
  File.write(File.join(dir, "text.txt"), Array.new(3000) { words.sample(random) }.join(" "))
  dir
end

private def settings(dir : String) : Finetune::Settings
  Finetune::Settings.new.tap do |run|
    run.model, run.data, run.heads = dir, File.join(dir, "text.txt"), 4
    run.batch, run.context, run.steps, run.warmup = 2, 16, 6, 2
    run.lr, run.min_lr, run.eval_batches, run.sample_length = 1e-2, 1e-3, 2, 5
    run.dropout = 0.1
  end
end

private def bits(trainer : Finetune::Trainer) : Array(Array(Float32))
  trainer.model.params.map(&.value.to_a(Float32)) + trainer.optimizer.state.to_a.sort_by(&.[0]).map(&.[1].to_a(Float32))
end

describe Finetune::Corpus do
  it "writes token ids and reads windows of them back" do
    path = File.tempname("tokens", ".bin")
    Finetune::Corpus.write(path, [0, 1, 50256, 65535, 7])
    corpus = Finetune::Corpus.new(path)
    corpus.size.should eq 5
    corpus.window(1, 3).should eq [1, 50256, 65535]
    expect_raises(IndexError) { corpus.window(3, 3) }
    expect_raises(ArgumentError) { Finetune::Corpus.write(path, [65536]) }
  ensure
    File.delete(path) if path && File.exists?(path)
  end
end

describe Finetune::Settings do
  it "warms up linearly, then decays along a cosine to the minimum" do
    s = Finetune::Settings.new
    s.lr, s.min_lr, s.warmup, s.steps = 1.0, 0.1, 4, 14
    s.lr_at(0).should eq 0.25
    s.lr_at(3).should eq 1.0
    s.lr_at(4).should eq 1.0
    s.lr_at(13).should be_close(0.1, 1e-12)
    s.lr_at(9).should be < s.lr_at(5)
  end
end

describe Finetune::Trainer do
  it "makes each step's batch from the seed and the step alone" do
    dir = workspace
    trainer = Finetune::Trainer.new(File.join(dir, "run"), settings(dir))
    a, b = trainer.batch(3), trainer.batch(3)
    a[0].to_a(Int32).should eq b[0].to_a(Int32)
    trainer.batch(4)[0].to_a(Int32).should_not eq a[0].to_a(Int32)
    inputs, targets = a
    inputs.shape.should eq [2, 16]
    targets.to_a(Int32)[0, 15].should eq inputs.to_a(Int32)[1, 15]
  ensure
    FileUtils.rm_rf(dir) if dir
  end

  it "resumes from a checkpoint with the bits of a run that never stopped" do
    dir = workspace
    straight = Finetune::Trainer.new(File.join(dir, "straight"), settings(dir))
    6.times { straight.step }

    first = Finetune::Trainer.new(File.join(dir, "resumed"), settings(dir))
    3.times { first.step }
    first.checkpoint
    resumed = Finetune::Trainer.new(File.join(dir, "resumed"), Finetune::Settings.new)
    resumed.progress.step.should eq 3
    resumed.settings.batch.should eq 2
    3.times { resumed.step }

    resumed.done?.should be_true
    bits(resumed).should eq bits(straight)
  ensure
    FileUtils.rm_rf(dir) if dir
  end

  it "lowers the training loss on its text" do
    dir = workspace
    trainer = Finetune::Trainer.new(File.join(dir, "run"), settings(dir).tap(&.steps=(30)))
    before = trainer.validation_loss
    30.times { trainer.step }
    trainer.validation_loss.should be < before
  ensure
    FileUtils.rm_rf(dir) if dir
  end

  it "keeps the best validation loss, exports it, and stops after its patience" do
    dir = workspace
    run = File.join(dir, "run")
    trainer = Finetune::Trainer.new(run, settings(dir).tap(&.patience=(2)))
    trainer.record(3.0).should be_false
    File.exists?(File.join(run, "best", "model.safetensors")).should be_true
    File.exists?(File.join(run, "best", "vocab.json")).should be_true
    trainer.record(2.5).should be_false
    trainer.record(2.6).should be_false
    trainer.record(2.7).should be_true
    trainer.progress.best.should eq 2.5
    GPT2.load(File.join(run, "best", "model.safetensors"), heads: 4).config.layers.should eq 2
  ensure
    FileUtils.rm_rf(dir) if dir
  end
end
