require "./tensomite/errors"
require "./tensomite/dtype"
require "./tensomite/view"
require "./tensomite/device"
require "./tensomite/node"
require "./tensomite/tensor"
require "./tensomite/tensor/*"
require "./tensomite/scheduler"
require "./tensomite/devices/host"
require "./tensomite/devices/cpu"
require "./tensomite/devices/sim_discrete"

module Tensomite
  # :nodoc:
  VERSION = {{ `shards version #{__DIR__}`.chomp.stringify }}
end
