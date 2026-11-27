module Tensomite
  # An allocation owned by a device. Only its device can read or write it.
  #
  # Its memory goes back to the device when the buffer is collected, or earlier
  # through `#release`. Until then it counts in its device's `Device#live_bytes`,
  # unless *counted* is false, as for a buffer over a file mapping.
  abstract class Buffer
    getter device : Device
    getter bytesize : Index
    @released = false

    def initialize(@device, bytesize : Int, @counted : Bool = true)
      @bytesize = bytesize.to_i64
      @device.track_live(@bytesize) if @counted
    end

    # Returns whether `#release` has given the memory back.
    def released? : Bool
      @released
    end

    # Gives the memory back to the device now, rather than when the buffer is
    # collected. The buffer must not be read or written afterwards.
    def release : Nil
      return if @released
      @released = true
      @device.track_live(-bytesize) if @counted
      give_back
    end

    def finalize
      return if @released
      @device.track_live(-bytesize) if @counted
      give_back
    end

    # Returns the memory to the device, which reuses or frees it.
    protected abstract def give_back : Nil
  end

  # A place where buffers live and kernels run.
  #
  # The interface always speaks in transfers. A device with unified memory may also
  # return a host view of a buffer through `#map`, which makes transfers optional.
  abstract class Device
    @live_bytes = Atomic(Int64).new(0)
    @peak_live_bytes = Atomic(Int64).new(0)

    # Returns the bytes held by this device's buffers that are not yet released
    # or collected, as PyTorch's `memory_allocated` gives them: memory kept in a
    # pool for reuse, and buffers over file mappings, are not counted.
    def live_bytes : Int64
      @live_bytes.get
    end

    # Returns the most `#live_bytes` has been since the device was created or
    # `#reset_peak` was last called, as PyTorch's `max_memory_allocated` does,
    # including peaks inside a single `Tensor.realize`.
    #
    # ```
    # device.reset_peak
    # optimizer.step(loss)
    # device.peak_live_bytes # the step's high-water mark
    # ```
    def peak_live_bytes : Int64
      @peak_live_bytes.get
    end

    # Sets the peak to the current `#live_bytes`.
    def reset_peak : Nil
      @peak_live_bytes.set(@live_bytes.get)
    end

    # :nodoc:
    # Adds *delta* to `#live_bytes` and raises the peak to it if it is higher.
    # Called from finalizers on any thread, so it never allocates or waits.
    def track_live(delta : Int64) : Nil
      live = @live_bytes.add(delta) + delta
      return unless delta > 0
      peak = @peak_live_bytes.get
      while live > peak
        peak, swapped = @peak_live_bytes.compare_and_set(peak, live)
        break if swapped
      end
    end

    # Returns a new buffer of *bytesize* bytes, contents undefined.
    abstract def alloc(bytesize : Int) : Buffer

    # Copies *data* from the host into *buffer*.
    abstract def upload(buffer : Buffer, data : Memory) : Nil

    # :ditto:
    def upload(buffer : Buffer, data : Bytes) : Nil
      upload(buffer, Memory.new(data))
    end

    # Copies *buffer* into host memory *data*.
    abstract def download(buffer : Buffer, data : Memory) : Nil

    # :ditto:
    def download(buffer : Buffer, data : Bytes) : Nil
      download(buffer, Memory.new(data))
    end

    # Returns a host view of *buffer*, or `nil` when the device memory is not host-visible.
    abstract def map(buffer : Buffer) : Memory?

    # Computes *node* into *output*, a contiguous buffer of `node.bytesize`, reading its sources' buffers.
    abstract def execute(node : Node, output : Buffer) : Nil

    # Blocks until all work submitted to the device has finished.
    abstract def sync : Nil

    # Returns true when a result's bits never depend on the shape of the computation
    # around it, such as batch size or prompt chunking (level 2 of Decision K).
    abstract def shape_invariant? : Bool

    # Returns true when the device computes `Op::Attention` itself, so that
    # `Tensor.attention` makes one; host devices compute attention as its
    # composite.
    def fuses_attention? : Bool
      false
    end
  end
end
