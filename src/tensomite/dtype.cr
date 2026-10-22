module Tensomite
  # The element type of a tensor, known at runtime.
  enum DType
    F32
    I32

    # Returns the number of bytes that hold *count* elements.
    #
    # Every byte-size calculation goes through here, because block formats store a
    # group of elements in a fixed number of bytes rather than a size per element.
    def bytes_for(count : Int32) : Int32
      case self
      in F32, I32 then count * 4
      end
    end

    # Returns true for floating-point types.
    def float? : Bool
      f32?
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
