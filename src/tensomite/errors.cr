module Tensomite
  # Base class for all errors raised by Tensomite.
  class Error < Exception; end

  # Raised when shapes are incompatible, raised as soon as the offending node is built.
  class ShapeError < Error; end

  # Raised when dtypes are incompatible, or when host access asks for the wrong type.
  class DTypeError < Error; end

  # Raised when an operation mixes tensors on different devices.
  class DeviceError < Error; end
end
