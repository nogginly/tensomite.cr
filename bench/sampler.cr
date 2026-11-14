# Times `Sampler#pick` on a row of GPT-2's 50,257 logits: greedy, top-k 40 at
# temperature 0.8, as `apps/server` and `apps/finetune` sample, and the whole
# vocabulary at temperature 1, the server's default.
#
#     crystal run --release bench/sampler.cr
require "../src/tensomite"

include Tensomite

VOCAB = 50_257
ROWS  =     64

rng = Random.new(7)
rows = Array.new(ROWS) { Array.new(VOCAB) { (rng.rand * 20 - 10).to_f32 } }

puts "sampler|µs per pick"
puts "-------|-----------"
{"greedy"                     => Sampler.greedy,
 "top_k 40, temperature 0.8"  => Sampler.new(temperature: 0.8, top_k: 40, seed: 1),
 "every token, temperature 1" => Sampler.new(temperature: 1.0, seed: 1)}.each do |label, sampler|
  rows.each { |row| sampler.pick(row) }
  best = Float64::MAX
  5.times do
    start = Time.instant
    rows.each { |row| sampler.pick(row) }
    best = Math.min(best, (Time.instant - start).total_microseconds / ROWS)
  end
  puts "#{label}|#{best.round(1)}"
end
