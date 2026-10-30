module Tensomite
  class CUDA < HostDevice
    class Profile
      # Prints, as Markdown tables, where *wall* seconds of *runs* runs of a
      # workload went, per *unit*: first by *normal*, a profile taken as the
      # workload ran, then the GPU time per op, by op and then by op and shape,
      # from *timed*, a profile with `per_kernel` and `shapes` of another run.
      # See `ProfileReport`.
      def self.report(io : IO, device : String, unit : String, runs : Int32, wall : Float64,
                      normal : Profile, timed : Profile, top : Int32 = 20) : Nil
        parts = [
          {"launching kernels", normal.launch},
          {"allocating buffers", normal.alloc},
          {"copying to and from the host", normal.transfer},
          {"waiting for the GPU", normal.wait},
        ]
        counts = [
          "launches: #{normal.launches // runs}; uploads: #{normal.uploads // runs}; downloads: #{normal.downloads // runs}",
          "allocations: #{normal.allocations // runs}; not from the pool: #{normal.pool_misses // runs}; freed: #{normal.frees // runs}",
        ]
        io << ProfileReport.new(device, unit, runs, wall, parts, counts, normal.host_ops, timed.kernels, top)
      end
    end
  end
end
