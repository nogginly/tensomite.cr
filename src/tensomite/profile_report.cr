module Tensomite
  # Prints, as Markdown tables, where a GPU device's time went over *runs* runs
  # of a workload, per *unit*. `CUDA::Profile.report` and `Metal::Profile.report`
  # fill it from their profiles.
  #
  # - *wall* is the wall seconds of a run taken as normal, and *parts* the
  #   seconds it spent in each named part of the device's work; the rest of the
  #   wall time, after the host ops, is reported as everything else.
  # - *counts* are lines of counters, printed as they are.
  # - *host_ops* holds the count and wall seconds of each op run on the host.
  # - *kernels* holds the count and GPU seconds of each op from another run that
  #   timed each op alone, keyed by its name and shapes, as
  #   `MatMul [64,768]·[768,2304]`. They are listed by op, then the *top*
  #   largest by op and shape, a matmul with its rate: the weights read per
  #   second when it has one row, as in decoding a token, and the flops
  #   otherwise.
  struct ProfileReport
    alias Totals = Hash(String, {Int32, Float64})

    def initialize(@device : String, @unit : String, @runs : Int32, @wall : Float64,
                   @parts : Array({String, Float64}), @counts : Array(String),
                   @host_ops : Totals, @kernels : Totals, @top : Int32 = 20)
    end

    def to_s(io : IO) : Nil
      host = @host_ops.values.sum(&.[1])
      io.puts "Per #{@unit} (#{@device}):"
      io.puts
      io.puts "part|ms"
      io.puts "----|--"
      io.puts "wall|#{ms(@wall)}"
      @parts.each { |name, seconds| io.puts "#{name}|#{ms(seconds)}" }
      io.puts "host ops|#{ms(host)}"
      io.puts "everything else|#{ms(@wall - @parts.sum(&.[1]) - host)}"
      io.puts
      @counts.each { |line| io.puts line }
      @host_ops.each { |name, (count, seconds)| io.puts "host op #{name}: #{per_run(count)} each, #{ms(seconds)} ms" }

      gpu = @kernels.values.sum(&.[1])
      io.puts
      io.puts "GPU time per #{@unit}, each op timed alone (total #{ms(gpu)} ms, #{(gpu / @wall * 100).round.to_i}% of the first run's wall time), by op:"
      io.puts
      io.puts "op|count|GPU ms|share"
      io.puts "--|-----|------|-----"
      by_op = Totals.new({0, 0.0})
      @kernels.each do |name, (count, seconds)|
        op = name.split(' ', 2).first
        total = by_op[op]
        by_op[op] = {total[0] + count, total[1] + seconds}
      end
      largest(by_op).each do |op, (count, seconds)|
        io.puts "#{op}|#{per_run(count)}|#{ms(seconds)}|#{(seconds / gpu * 100).round.to_i}%"
      end
      io.puts
      io.puts "By op and shape, the #{@top} largest:"
      io.puts
      io.puts "op|count|GPU ms|share|GB/s or GFLOP/s"
      io.puts "--|-----|------|-----|---------------"
      largest(@kernels).first(@top).each do |name, (count, seconds)|
        io.puts "#{name}|#{per_run(count)}|#{ms(seconds)}|#{(seconds / gpu * 100).round.to_i}%|#{rate(name, count, seconds)}"
      end
    end

    private def ms(seconds : Float64) : String
      "%.3f" % (seconds * 1000 / @runs)
    end

    private def per_run(count : Int32) : String
      "%.1f" % (count / @runs)
    end

    private def largest(totals : Totals) : Array({String, {Int32, Float64}})
      totals.to_a.sort_by { |(_, (_, seconds))| -seconds }
    end

    # Returns a matmul's rate from its label, `MatMul [..., M, K]·[..., K, N]`:
    # 2·M·K·N flops per batch matrix, or with one row K·N weights read.
    private def rate(name : String, count : Int32, seconds : Float64) : String
      return "" unless name.starts_with?("MatMul") && (dims = name.scan(/\d+/).map(&.[0].to_i)).size >= 4
      a, b = dims[0, dims.size // 2], dims[dims.size // 2..]
      if a[-2] == 1
        "%.1f GB/s" % (4.0 * b[-2] * b[-1] * b[0...-2].product(1) * count / seconds / 1e9)
      else
        "%.1f GFLOP/s" % (2.0 * a.product(1) * b.last * count / seconds / 1e9)
      end
    end
  end
end
