module Tensomite
  # The reference device: host kernels on host-visible memory. Every other device
  # is tested against it.
  class CPU < HostDevice
    # Returns the shared CPU device.
    class_getter default : CPU { CPU.new }

    def map(buffer : Buffer) : Memory?
      own(buffer).memory
    end
  end
end
