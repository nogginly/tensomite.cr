module Tensomite
  # A test device that behaves like a discrete GPU: host kernels on memory the host
  # cannot see. `#map` always returns `nil`, so data moves only through `#upload`
  # and `#download`, which are counted.
  #
  # Running a suite on it finds code that assumes it can read a buffer directly.
  class SimDiscrete < HostDevice
    # Returns the shared simulated device.
    class_getter default : SimDiscrete { SimDiscrete.new }

    # Returns the number of uploads since creation.
    getter uploads = 0

    # Returns the number of downloads since creation.
    getter downloads = 0

    def map(buffer : Buffer) : Bytes?
      own(buffer)
      nil
    end

    def upload(buffer : Buffer, data : Bytes) : Nil
      @uploads += 1
      super
    end

    def download(buffer : Buffer, data : Bytes) : Nil
      @downloads += 1
      super
    end
  end
end
