module Tensomite
  # The element type of a tensor, known at runtime.
  enum DType
    F32
    I32

    # Returns the size of one element in bytes.
    def bytesize : Int32
      case self
      in F32, I32 then 4
      end
    end

    # Returns the `DType` matching a Crystal type, e.g. `DType.of(Float32) # => DType::F32`.
    def self.of(type : Float32.class) : DType
      F32
    end

    # :ditto:
    def self.of(type : Int32.class) : DType
      I32
    end
  end
end
