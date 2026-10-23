module Tensomite
  # A read-only, private memory mapping of a whole file, unmapped when collected.
  #
  # Buffers over parts of the mapping keep it alive (see `HostBuffer`), so it lasts
  # as long as any tensor that reads it.
  class MappedFile
    getter path : String
    getter pointer : Pointer(UInt8)
    getter size : Int64

    # Maps the file at *path*; raises `Error` when the system refuses.
    def initialize(@path : String)
      file = File.open(path)
      @size = file.size.to_i64
      address = LibC.mmap(nil, @size, LibC::PROT_READ, LibC::MAP_PRIVATE, file.fd, 0)
      file.close
      raise Error.new("cannot map #{path}: #{Errno.value.message}") if address == LibC::MAP_FAILED
      @pointer = address.as(UInt8*)
    end

    # Returns the *length* bytes at *offset*, without copying.
    def bytes(offset : Int64, length : Int32) : Bytes
      unless 0 <= offset && offset + length <= size
        raise Error.new("#{path}: bytes #{offset}...#{offset + length} lie outside #{size}")
      end
      Bytes.new(pointer + offset, length, read_only: true)
    end

    def finalize
      LibC.munmap(@pointer.as(Void*), @size)
    end
  end
end
