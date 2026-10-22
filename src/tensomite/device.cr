module Tensomite
  # An allocation owned by a device. Only its device can read or write it.
  abstract class Buffer
    getter bytesize : Int32

    def initialize(@bytesize)
    end
  end

  # A place where buffers live and kernels run.
  #
  # The interface always speaks in transfers. A device with unified memory may also
  # return a host view of a buffer through `#map`, which makes transfers optional.
  abstract class Device
    # Returns a new buffer of *bytesize* bytes, contents undefined.
    abstract def alloc(bytesize : Int32) : Buffer

    # Copies *data* from the host into *buffer*.
    abstract def upload(buffer : Buffer, data : Bytes) : Nil

    # Copies *buffer* into host memory *data*.
    abstract def download(buffer : Buffer, data : Bytes) : Nil

    # Returns a host view of *buffer*, or `nil` when the device memory is not host-visible.
    abstract def map(buffer : Buffer) : Bytes?

    # Computes *node* into *output*, a contiguous buffer of `node.bytesize`, reading its sources' buffers.
    abstract def execute(node : Node, output : Buffer) : Nil

    # Blocks until all work submitted to the device has finished.
    abstract def sync : Nil
  end
end
