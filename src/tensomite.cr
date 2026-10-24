require "./tensomite/errors"
require "./tensomite/dtype"
require "./tensomite/view"
require "./tensomite/device"
require "./tensomite/buffer_pool"
require "./tensomite/maths"
require "./tensomite/scope"
require "./tensomite/node"
require "./tensomite/tensor"
require "./tensomite/tensor/*"
require "./tensomite/scheduler"
require "./tensomite/autograd"
require "./tensomite/diagram"
require "./tensomite/kernel"
require "./tensomite/kernels"
require "./tensomite/optim"
require "./tensomite/devices/host"
require "./tensomite/devices/cpu"
require "./tensomite/devices/sim_discrete"
require "./tensomite/mapped_file"
require "./tensomite/safetensors"
require "./tensomite/bpe"
require "./tensomite/sampler"
require "./tensomite/models/gpt2"

module Tensomite
  # :nodoc:
  VERSION = {{ `shards version #{__DIR__}`.chomp.stringify }}
end
