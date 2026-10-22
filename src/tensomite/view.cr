module Tensomite
  # Describes how a logical shape maps onto a flat run of elements, through strides and an offset.
  #
  # Views are pure metadata: reshaping, permuting and broadcasting produce new views
  # and never touch data. Strides and offset are counted in elements, not bytes.
  #
  # ```
  # v = View.contiguous([2, 3]) # strides [3, 1]
  # v.permute([1, 0])           # shape [3, 2], strides [1, 3]
  # v.expand([4, 2, 3])         # shape [4, 2, 3], strides [0, 3, 1]
  # ```
  struct View
    getter shape : Array(Int32)
    getter strides : Array(Int32)
    getter offset : Int32

    def initialize(shape : Array(Int32), strides : Array(Int32), @offset = 0)
      raise ShapeError.new("shape #{shape} and strides #{strides} differ in rank") unless shape.size == strides.size
      raise ShapeError.new("negative dimension in #{shape}") if shape.any?(&.negative?)
      @shape = shape.dup
      @strides = strides.dup
    end

    # Returns a row-major view of *shape* starting at element 0.
    def self.contiguous(shape : Array(Int32)) : View
      new(shape, row_major_strides(shape))
    end

    # Returns the row-major strides for *shape*.
    def self.row_major_strides(shape : Array(Int32)) : Array(Int32)
      strides = Array(Int32).new(shape.size, 1)
      (shape.size - 2).downto(0) { |i| strides[i] = strides[i + 1] * shape[i + 1] }
      strides
    end

    # Returns the shape that *a* and *b* broadcast to, following NumPy rules.
    def self.broadcast_shape(a : Array(Int32), b : Array(Int32)) : Array(Int32)
      rank = {a.size, b.size}.max
      Array(Int32).new(rank) do |i|
        da = i < rank - a.size ? 1 : a[i - (rank - a.size)]
        db = i < rank - b.size ? 1 : b[i - (rank - b.size)]
        unless da == db || da == 1 || db == 1
          raise ShapeError.new("shapes #{a} and #{b} cannot be broadcast")
        end
        da == 1 ? db : da
      end
    end

    # Returns the number of dimensions.
    def ndim : Int32
      shape.size
    end

    # Returns the number of elements; a rank-0 view has one.
    def size : Int32
      shape.product(1)
    end

    # Returns true when the view walks its elements in row-major order from element 0.
    def contiguous? : Bool
      offset == 0 && strides == View.row_major_strides(shape)
    end

    # Returns a view with dimensions reordered so that new dimension `i` is old dimension `axes[i]`.
    def permute(axes : Array(Int32)) : View
      unless axes.size == ndim && axes.sort == (0...ndim).to_a
        raise ShapeError.new("#{axes} is not a permutation of #{ndim} axes")
      end
      View.new(axes.map { |a| shape[a] }, axes.map { |a| strides[a] }, offset)
    end

    # Returns a view with dimensions *a* and *b* swapped; negative axes count from the end.
    def transpose(a : Int32 = -2, b : Int32 = -1) : View
      axes = (0...ndim).to_a
      a, b = normalize_axis(a), normalize_axis(b)
      axes[a], axes[b] = axes[b], axes[a]
      permute(axes)
    end

    # Returns a view broadcast to *target*: size-1 and missing leading dimensions get stride 0.
    def expand(target : Array(Int32)) : View
      unless View.broadcast_shape(shape, target) == target
        raise ShapeError.new("cannot expand #{shape} to #{target}")
      end
      lead = target.size - ndim
      new_strides = Array(Int32).new(target.size) do |i|
        i < lead || (shape[i - lead] == 1 && target[i] != 1) ? 0 : strides[i - lead]
      end
      View.new(target, new_strides, offset)
    end

    # Returns a view of indices `start...stop` along *axis*, sharing the same elements.
    def slice(axis : Int32, start : Int32, stop : Int32) : View
      axis = normalize_axis(axis)
      unless 0 <= start <= stop <= shape[axis]
        raise ShapeError.new("slice #{start}...#{stop} out of range for axis #{axis} of #{shape}")
      end
      new_shape = shape.dup
      new_shape[axis] = stop - start
      View.new(new_shape, strides, offset + start * strides[axis])
    end

    # Returns a view of the same elements with a new shape, or `nil` when the view
    # is not contiguous and the data must be copied first.
    def reshape(target : Array(Int32)) : View?
      unless target.product(1) == size
        raise ShapeError.new("cannot reshape #{shape} (#{size} elements) to #{target}")
      end
      View.contiguous(target) if contiguous?
    end

    # Returns the storage position of the element at row-major index *flat*.
    def offset_of(flat : Int32) : Int32
      pos = offset
      (ndim - 1).downto(0) do |i|
        pos += (flat % shape[i]) * strides[i]
        flat //= shape[i]
      end
      pos
    end

    # Returns *axis* as a non-negative index, counting negative values from the end.
    def normalize_axis(axis : Int32) : Int32
      norm = axis < 0 ? axis + ndim : axis
      raise ShapeError.new("axis #{axis} out of range for rank #{ndim}") unless 0 <= norm < ndim
      norm
    end
  end
end
