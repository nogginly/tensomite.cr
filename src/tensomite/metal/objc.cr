module Tensomite
  @[Link("objc")]
  @[Link(framework: "Foundation")]
  lib LibObjC
    fun objc_getClass(name : UInt8*) : Void*
    fun sel_registerName(name : UInt8*) : Void*
    fun objc_msgSend(receiver : Void*, selector : Void*) : Void*
    fun objc_autoreleasePoolPush : Void*
    fun objc_autoreleasePoolPop(pool : Void*) : Void
  end

  # Calls into the Objective-C runtime.
  #
  # Every message goes through `objc_msgSend`, which must be called with the exact C
  # signature of the method it reaches. `.send` casts it to a `Proc` typed by the
  # arguments it is given and the return type it is asked for, so each call site
  # states its signature:
  #
  # ```
  # length = ObjC.send(UInt64, buffer, "length")
  # ObjC.send(Nil, encoder, "setBuffer:offset:atIndex:", buffer, 0_u64, 2_u64)
  # ```
  #
  # Pass `NSUInteger` as `UInt64`, `BOOL` as `UInt32` (read a returned one as `UInt8`), objects as `Void*`, and C
  # structs larger than 16 bytes (such as `MTLSize`) as a pointer to a copy, which
  # is how arm64 passes them. That last rule makes these bindings Apple Silicon only.
  module ObjC
    private MSG_SEND = (->LibObjC.objc_msgSend).pointer
    @@selectors = {} of String => Void*

    # Sends *selector* with *args* to *receiver* and returns its result as *type*.
    def self.send(type : R.class, receiver : Void*, selector : String, *args : *T) : R forall R, T
      Proc(Void*, Void*, *T, R).new(MSG_SEND, Pointer(Void).null).call(receiver, sel(selector), *args)
    end

    # Returns the class named *name*; raises `DeviceError` when it does not exist.
    def self.cls(name : String) : Void*
      cls = LibObjC.objc_getClass(name)
      raise DeviceError.new("no Objective-C class #{name}") if cls.null?
      cls
    end

    # Returns the selector named *name*, registered once.
    def self.sel(name : String) : Void*
      @@selectors[name] ||= LibObjC.sel_registerName(name)
    end

    # Returns an autoreleased `NSString` holding *string*.
    def self.string(string : String) : Void*
      send(Pointer(Void), cls("NSString"), "stringWithUTF8String:", string.to_unsafe)
    end

    # Returns the text of an `NSError`, or of an `NSString` when *object* is one.
    def self.describe(object : Void*) : String
      return "unknown error" if object.null?
      error = send(UInt8, object, "respondsToSelector:", sel("localizedDescription")) != 0
      text = error ? send(Pointer(Void), object, "localizedDescription") : object
      String.new(send(Pointer(UInt8), text, "UTF8String"))
    end

    # Runs the block inside an autorelease pool, so objects the runtime autoreleases
    # during it are freed when it ends, and returns the block's value.
    def self.autoreleasepool(& : -> T) : T forall T
      pool = LibObjC.objc_autoreleasePoolPush
      begin
        yield
      ensure
        LibObjC.objc_autoreleasePoolPop(pool)
      end
    end

    # Takes ownership of *object*, which then survives its autorelease pool.
    def self.retain(object : Void*) : Void*
      send(Pointer(Void), object, "retain")
    end

    # Gives up ownership of *object*.
    def self.release(object : Void*) : Nil
      send(Nil, object, "release") unless object.null?
    end
  end
end
