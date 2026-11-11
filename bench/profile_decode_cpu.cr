# Breaks down where a model's time goes on a host device, per decoded token:
# GPT-2, or a Llama-family model, quantised or not.
#
#     crystal run --release bench/profile_decode_cpu.cr -- models/smollm2-1.7b-instruct --quantize Q4_0
#     crystal run --release bench/profile_decode_cpu.cr -- models/smollm2-1.7b-instruct --quantize Q4_0 --device cpu
#
# `--device` is `blas` (the default) or `cpu`. `--prompt N` and `--decode N` set
# the prompt's length (16 by default) and the tokens decoded after it (8 by
# default): fewer than the GPU profilers', since host kernels are slower. The
# workload runs twice: once for the wall time per token, then with each op
# timed, by op.
require "../src/tensomite/blas"
require "./profile_load"

include Tensomite

path, heads, _, quantize = ProfileBench.arguments
device = ProfileBench.option("--device") == "cpu" ? CPU.default.as(HostDevice) : BLAS.default.as(HostDevice)
prompt = ProfileBench.option("--prompt").try(&.to_i) || 16
decode = ProfileBench.option("--decode").try(&.to_i) || 8

# Profiles *model* on *device* and prints the report.
def profile(model : M, device : HostDevice, prompt : Int32, decode : Int32) : Nil forall M
  wall = ProfileBench.run(model, nil, prompt, decode) { } / decode * 1000
  times = {} of String => {Int32, Float64}
  ProfileBench.run(model, nil, prompt, decode) { |measuring| device.op_times = measuring ? times : nil }
  puts "Per decoded token (#{device.class}):\n\n"
  puts "part|ms\n----|--"
  puts "wall|#{wall.round(3)}"
  puts "tokens/s|#{(1000 / wall).round(1)}"
  total = times.sum { |_, (_, ms)| ms } / decode
  puts "\nOps, each timed (total #{total.round(3)} ms, #{(total / wall * 100).round.to_i}% of the wall time):\n\n"
  puts "op|count|ms|share\n--|-----|--|-----"
  times.to_a.sort_by { |_, (_, ms)| -ms }.each do |op, (count, ms)|
    per = ms / decode
    puts "#{op}|#{(count / decode).round(1)}|#{per.round(3)}|#{(per / total * 100).round.to_i}%"
  end
end

if ProfileBench.llama?(path)
  profile(ProfileBench.quantize(Llama.load(path, device), quantize), device, prompt, decode)
else
  profile(ProfileBench.quantize(GPT2.load(path, heads, device), quantize), device, prompt, decode)
end
