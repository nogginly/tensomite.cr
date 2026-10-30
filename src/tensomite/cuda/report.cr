module Tensomite
  class CUDA < HostDevice
    class Profile
      # Prints, as Markdown tables, where *wall* seconds of *runs* runs of a
      # workload went, per *unit*: first by *normal*, a profile taken as the
      # workload ran, then the GPU time per op, by op and then by op and shape,
      # from *timed*, a profile with `per_kernel` and `shapes` of another run.
      #
      # A matmul by shape also shows its rate: the weights read per second when
      # it has one row, as in decoding a token, and the flops otherwise.
      def self.report(io : IO, device : String, unit : String, runs : Int32, wall : Float64,
                      normal : Profile, timed : Profile, top : Int32 = 20) : Nil
        ms = ->(seconds : Float64) { "%.3f" % (seconds * 1000 / runs) }
        per_run = ->(count : Int32) { "%.1f" % (count / runs) }
        host = normal.host_ops.values.sum(&.[1])
        other = wall - normal.launch - normal.alloc - normal.transfer - normal.wait - host
        io.puts "Per #{unit} (#{device}):"
        io.puts
        io.puts "part|ms"
        io.puts "----|--"
        io.puts "wall|#{ms.call(wall)}"
        io.puts "launching kernels|#{ms.call(normal.launch)}"
        io.puts "allocating buffers|#{ms.call(normal.alloc)}"
        io.puts "copying to and from the host|#{ms.call(normal.transfer)}"
        io.puts "waiting for the GPU|#{ms.call(normal.wait)}"
        io.puts "host ops|#{ms.call(host)}"
        io.puts "everything else|#{ms.call(other)}"
        io.puts
        io.puts "launches: #{normal.launches // runs}; uploads: #{normal.uploads // runs}; downloads: #{normal.downloads // runs}"
        io.puts "allocations: #{normal.allocations // runs}; not from the pool: #{normal.pool_misses // runs}; freed: #{normal.frees // runs}"
        normal.host_ops.each { |name, (count, seconds)| io.puts "host op #{name}: #{per_run.call(count)} each, #{ms.call(seconds)} ms" }

        gpu = timed.kernels.values.sum(&.[1])
        io.puts
        io.puts "GPU time per #{unit}, each op timed alone (total #{ms.call(gpu)} ms, #{(gpu / wall * 100).round}% of the first run's wall time), by op:"
        io.puts
        io.puts "op|count|GPU ms|share"
        io.puts "--|-----|------|-----"
        by_op = Hash(String, {Int32, Float64}).new({0, 0.0})
        timed.kernels.each do |name, (count, seconds)|
          op = name.split(' ', 2).first
          total = by_op[op]
          by_op[op] = {total[0] + count, total[1] + seconds}
        end
        by_op.to_a.sort_by { |(_, (_, seconds))| -seconds }.each do |op, (count, seconds)|
          io.puts "#{op}|#{per_run.call(count)}|#{ms.call(seconds)}|#{(seconds / gpu * 100).round}%"
        end
        io.puts
        io.puts "By op and shape, the #{top} largest:"
        io.puts
        io.puts "op|count|GPU ms|share|GB/s or GFLOP/s"
        io.puts "--|-----|------|-----|---------------"
        timed.kernels.to_a.sort_by { |(_, (_, seconds))| -seconds }.first(top).each do |name, (count, seconds)|
          io.puts "#{name}|#{per_run.call(count)}|#{ms.call(seconds)}|#{(seconds / gpu * 100).round}%|#{rate(name, count, seconds)}"
        end
      end

      # Returns a matmul's rate from its label, `MatMul [..., M, K]·[..., K, N]`:
      # 2·M·K·N flops per batch matrix, or with one row K·N weights read.
      private def self.rate(name : String, count : Int32, seconds : Float64) : String
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
end
