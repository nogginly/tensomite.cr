require "./tensomite/errors"
require "./tensomite/dtype"
require "./tensomite/view"
require "./tensomite/device"
require "./tensomite/node"
require "./tensomite/tensor"
require "./tensomite/scheduler"
require "./tensomite/devices/cpu"

module Tensomite
  # :nodoc:
  VERSION = {{ `shards version #{__DIR__}`.chomp.stringify }}
end
