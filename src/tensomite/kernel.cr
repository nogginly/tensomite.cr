module Tensomite
  # An elementwise expression fused into one kernel: one pass over memory on host
  # devices, one dispatch on Metal and one launch on CUDA. Define one with
  # `Tensomite.kernel`.
  #
  # A kernel's inputs are broadcast to a common shape, like a binary op's, and must
  # all be `F32` tensors on one device.
  abstract class Kernel
    # The name the kernel was defined with.
    getter name : String

    # The number of inputs.
    getter arity : Int32

    def initialize(@name, @arity)
    end

    # Returns the kernel applied to *inputs*, lazily.
    def call(*inputs : Tensor) : Tensor
      Tensor.kernel(self, inputs.to_a)
    end

    # Writes the kernel of each logical element of *views*, which read *srcs*, into
    # *dst*, walking the views in runs (see `View.run_length`).
    abstract def host(dst : Slice(Float32), srcs : Array(Slice(Float32)), views : Array(View)) : Nil

    # Returns the kernel as an MSL function named `#msl_name`, to compile after
    # `metal/prelude.metal`. Constants are written in full, and every operation
    # is its own statement, so the compiler cannot fuse a multiply and an add.
    abstract def msl : String

    # Returns the kernel as a CUDA C++ function named `#cuda_name`, declared
    # `extern "C" __global__`, to compile after `cuda/prelude.cu`. As with `#msl`,
    # every operation is its own statement; NVRTC runs with fused multiply-adds
    # off, so a multiply and an add each round.
    abstract def cuda : String

    # Returns the kernel that computes the gradient of input *index* from the
    # output's gradient and the inputs, or `nil` for a kernel that is itself a
    # gradient.
    abstract def gradient(index : Int32) : Kernel?

    def msl_name : String
      "k_#{name}"
    end

    def cuda_name : String
      "k_#{name}"
    end

    # Returns *value* as an MSL float literal that reads back as the same bits.
    protected def self.msl_float(value : Float32) : String
      raise Error.new("kernel constant #{value} has no MSL literal") unless value.finite?
      "#{value}f"
    end
  end

  # Defines an elementwise kernel named *name* from a block over its inputs, and
  # the constant `Tensomite::Kernels::NAME` (the name upper-cased) holding it.
  #
  # ```
  # Tensomite.kernel(:axpb) { |x, a, b| x * a + b }
  # Tensomite::Kernels::AXPB.call(x, a, b) # one kernel, not two
  # ```
  #
  # The block may use `+`, `-` (binary and unary), `*`, `/`, number literals,
  # constants such as `Math::PI`, and `tanh`, `exp`, `log` and `sqrt`, written as
  # `x.tanh`, `tanh(x)` or `Math.tanh(x)`.
  #
  # At compile time the macro flattens the block into a tape of single
  # operations, in Crystal's own order of evaluation, then generates:
  #
  # 1. a host loop that computes each operation in `Float32`, with the same
  #    functions (`Tensomite::Maths`, and `Math.sqrt`) as the host kernels, as the
  #    equivalent composite of primitive ops does, so the two give the same bits;
  # 2. MSL source with one statement per operation;
  # 3. one gradient kernel per input, by a reverse sweep over the tape.
  #
  # A subexpression without inputs, such as `Math.sqrt(2 / Math::PI)`, is
  # computed once in `Float64` and converted to `Float32`, as `Tensor.scalar`
  # converts a number operand.
  macro kernel(name, &block)
    {% args = block.args.map { |arg| "in_#{arg.id}" } %}
    {% arg_names = block.args.map(&.id.stringify) %}
    {% consts = [] of ASTNode %}
    {% statements = [] of ASTNode %}
    {% values = [] of ASTNode %}
    {% stack = [[block.body, false]] %}
    {% functions = %w[tanh exp log sqrt] %}

    # 1. Post-order walk with an explicit stack: each operation's operands are
    #    pushed onto `values` before it, and it pops them and pushes its result.
    {% for step in 0..2000 %}
      {% unless stack.empty? %}
        {% entry = stack.last %}
        {% stack = stack.size > 1 ? stack[0...-1] : [] of ASTNode %}
        {% node = entry[0] %}
        {% if node.is_a?(Expressions) && node.expressions.size == 1 %}
          {% stack << [node.expressions[0], entry[1]] %}
        {% elsif node.is_a?(Var) || (node.is_a?(Call) && node.receiver.is_a?(Nop) && node.args.empty? && arg_names.includes?(node.name.stringify)) %}
          {% values << "in_#{node.id}" %}
        {% elsif node.is_a?(NumberLiteral) %}
          {% consts << "(#{node}).to_f64" %}
          {% values << "c#{consts.size - 1}" %}
        {% elsif node.is_a?(Path) %}
          {% consts << "#{node}.to_f64" %}
          {% values << "c#{consts.size - 1}" %}
        {% elsif node.is_a?(Call) %}
          {% name_s = node.name.stringify %}
          # `Math.tanh(x)` names a function in a namespace; in `Math::PI / 2` the
          # path is an operand.
          {% namespaced = node.receiver.is_a?(Path) && functions.includes?(name_s) %}
          {% receiver_is_operand = !node.receiver.is_a?(Nop) && !namespaced %}
          {% operands = receiver_is_operand ? [node.receiver] + node.args : node.args %}
          {% op = nil %}
          {% if %w[+ - * /].includes?(name_s) && operands.size == 2 %}
            {% op = name_s %}
          {% elsif name_s == "-" && operands.size == 1 %}
            {% op = "neg" %}
          {% elsif functions.includes?(name_s) && operands.size == 1 %}
            {% op = name_s %}
          {% end %}
          {% raise "kernel #{name.id}: unsupported operation `#{node}`" unless op %}
          {% if !entry[1] %}
            {% stack << [node, true] %}
            {% for i in 1..operands.size %}{% stack << [operands[operands.size - i], false] %}{% end %}
          {% else %}
            {% n = operands.size %}
            {% ins = values[values.size - n..-1] %}
            {% values = values.size > n ? values[0...values.size - n] : [] of ASTNode %}
            {% if ins.all?(&.starts_with?("c")) %}
              # A subexpression of constants folds into one constant, in Float64.
              {% exprs = ins.map { |const| "(#{consts[const[1..-1].to_i].id})" } %}
              {% if op == "neg" %}
                {% consts << "(-#{exprs[0].id})" %}
              {% elsif functions.includes?(op) %}
                {% consts << "Math.#{op.id}#{exprs[0].id}" %}
              {% else %}
                {% consts << "(#{exprs[0].id} #{op.id} #{exprs[1].id})" %}
              {% end %}
              {% values << "c#{consts.size - 1}" %}
            {% else %}
              {% statements << ["t#{statements.size}", op, ins[0], ins.size > 1 ? ins[1] : ""] %}
              {% values << "t#{statements.size - 1}" %}
            {% end %}
          {% end %}
        {% else %}
          {% raise "kernel #{name.id}: unsupported expression `#{node}`" %}
        {% end %}
      {% end %}
    {% end %}
    {% raise "kernel #{name.id}: expression too large" unless stack.empty? %}
    {% result = values[0] %}

    # 2. For each input, a reverse sweep over the tape gives its gradient: each
    #    statement's output gradient spreads to its operands by its op's rule, and
    #    contributions to one value are added in the order the sweep meets them.
    {% gradients = [] of ASTNode %}
    {% for target in args %}
      {% backward = statements.map { |statement| statement } %}
      {% adjoint = {result => "g"} %}
      {% for k in 1..statements.size %}
        {% s = statements[statements.size - k] %}
        {% out = adjoint[s[0]] %}
        {% if out %}
          {% contributions = [] of ASTNode %}
          {% a = s[2] %}
          {% b = s[3] %}
          {% op = s[1] %}
          {% if op == "+" %}
            {% contributions << [a, out] %}
            {% contributions << [b, out] %}
          {% elsif op == "-" %}
            {% contributions << [a, out] %}
            {% unless b.starts_with?("c") %}
              {% backward << ["u#{backward.size}", "neg", out, ""] %}
              {% contributions << [b, "u#{backward.size - 1}"] %}
            {% end %}
          {% elsif op == "*" %}
            {% unless a.starts_with?("c") %}
              {% backward << ["u#{backward.size}", "*", out, b] %}
              {% contributions << [a, "u#{backward.size - 1}"] %}
            {% end %}
            {% unless b.starts_with?("c") %}
              {% backward << ["u#{backward.size}", "*", out, a] %}
              {% contributions << [b, "u#{backward.size - 1}"] %}
            {% end %}
          {% elsif op == "/" %}
            {% unless a.starts_with?("c") %}
              {% backward << ["u#{backward.size}", "/", out, b] %}
              {% contributions << [a, "u#{backward.size - 1}"] %}
            {% end %}
            {% unless b.starts_with?("c") %}
              {% backward << ["u#{backward.size}", "*", out, s[0]] %}
              {% backward << ["u#{backward.size}", "/", "u#{backward.size - 1}", b] %}
              {% backward << ["u#{backward.size}", "neg", "u#{backward.size - 1}", ""] %}
              {% contributions << [b, "u#{backward.size - 1}"] %}
            {% end %}
          {% elsif op == "neg" %}
            {% backward << ["u#{backward.size}", "neg", out, ""] %}
            {% contributions << [a, "u#{backward.size - 1}"] %}
          {% elsif op == "tanh" %}
            {% backward << ["u#{backward.size}", "*", s[0], s[0]] %}
            {% backward << ["u#{backward.size}", "-", "one", "u#{backward.size - 1}"] %}
            {% backward << ["u#{backward.size}", "*", out, "u#{backward.size - 1}"] %}
            {% contributions << [a, "u#{backward.size - 1}"] %}
          {% elsif op == "exp" %}
            {% backward << ["u#{backward.size}", "*", out, s[0]] %}
            {% contributions << [a, "u#{backward.size - 1}"] %}
          {% elsif op == "log" %}
            {% backward << ["u#{backward.size}", "/", out, a] %}
            {% contributions << [a, "u#{backward.size - 1}"] %}
          {% elsif op == "sqrt" %}
            {% backward << ["u#{backward.size}", "*", s[0], "two"] %}
            {% backward << ["u#{backward.size}", "/", out, "u#{backward.size - 1}"] %}
            {% contributions << [a, "u#{backward.size - 1}"] %}
          {% end %}
          {% for c in contributions %}
            {% unless c[0].starts_with?("c") %}
              {% if adjoint[c[0]] %}
                {% backward << ["u#{backward.size}", "+", adjoint[c[0]], c[1]] %}
                {% adjoint[c[0]] = "u#{backward.size - 1}" %}
              {% else %}
                {% adjoint[c[0]] = c[1] %}
              {% end %}
            {% end %}
          {% end %}
        {% end %}
      {% end %}
      {% gradients << [backward, adjoint[target] || "zero"] %}
    {% end %}

    {% class_name = name.id.stringify.camelcase.id %}
    module ::Tensomite::Kernels
      {% for gradient, i in gradients %}
        ::Tensomite.__kernel_class({{ class_name }}Gradient{{ i }}, {{ "#{name.id}_gradient#{i}" }},
          {{ ["g"] + args }}, {{ consts.empty? ? "[] of String".id : consts }}, {{ gradient[0] }}, {{ gradient[1] }}, nil)
      {% end %}
      ::Tensomite.__kernel_class({{ class_name }}, {{ name.id.stringify }}, {{ args }}, {{ consts.empty? ? "[] of String".id : consts }}, {{ statements.empty? ? "[] of String".id : statements }}, {{ result }},
        {{ (0...gradients.size).map { |i| "#{class_name}Gradient#{i}" } }})

      {{ name.id.stringify.upcase.id }} = {{ class_name }}.new
    end
  end

  # :nodoc:
  # Defines a `Kernel` subclass from a statement list made by `Tensomite.kernel`.
  macro __kernel_class(class_name, kernel_name, inputs, consts, statements, result, gradients)
    class {{ class_name }} < ::Tensomite::Kernel
      {% for c, i in consts %}
        C{{ i }} = ({{ c.id }}).to_f32
      {% end %}

      def initialize
        super({{ kernel_name }}, {{ inputs.size }})
      end

      # Returns the kernel's value for one element.
      def self.eval({{ inputs.map { |input| "#{input.id} : Float32".id }.splat }}) : Float32
        {% for s in statements %}
          {% operands = [s[2], s[3]].map { |v| v.starts_with?("c") ? "C#{v[1..-1].id}" : (v == "one" ? "1.0_f32" : (v == "two" ? "2.0_f32" : (v == "zero" ? "0.0_f32" : v))) } %}
          {% if s[1] == "neg" %}
            {{ s[0].id }} = -{{ operands[0].id }}
          {% elsif %w[tanh exp log sqrt].includes?(s[1]) %}
            {{ s[0].id }} = {{ (s[1] == "sqrt" ? "Math" : "::Tensomite::Maths").id }}.{{ s[1].id }}({{ operands[0].id }})
          {% else %}
            {{ s[0].id }} = {{ operands[0].id }} {{ s[1].id }} {{ operands[1].id }}
          {% end %}
        {% end %}
        {{ result.starts_with?("c") ? "C#{result[1..-1].id}".id : (result == "zero" ? "0.0_f32".id : result.id) }}
      end

      def host(dst : Slice(Float32), srcs : Array(Slice(Float32)), views : Array(::Tensomite::View)) : Nil
        return if dst.empty?
        run = ::Tensomite::View.run_length(dst.size, views)
        {% for input, i in inputs %}
          xs{{ i }}, stride{{ i }} = srcs[{{ i }}], views[{{ i }}].run_stride
        {% end %}
        unit = {{ inputs.map_with_index { |_, i| "stride#{i} == 1".id }.join(" && ").id }}
        (dst.size // run).times do |run_index|
          # Checks each run's extent once, then reads through pointers, so that
          # unit-stride runs vectorise.
          base = run_index * run
          output = dst.to_unsafe + base
          {% for input, i in inputs %}
            pos{{ i }} = views[{{ i }}].offset_of(base)
            ::Tensomite::HostDevice.check_run(pos{{ i }}, run, stride{{ i }}, xs{{ i }}.size) { "kernel #{name} source {{ i }} #{views[{{ i }}].inspect}" }
            p{{ i }} = xs{{ i }}.to_unsafe + pos{{ i }}
          {% end %}
          if unit
            run.times do |j|
              output[j] = self.class.eval({{ inputs.map_with_index { |_, i| "p#{i}[j]".id }.splat }})
            end
          else
            run.times do |j|
              output[j] = self.class.eval({{ inputs.map_with_index { |_, i| "p#{i}[j * stride#{i}]".id }.splat }})
            end
          end
        end
      end

      def msl : String
        String.build do |io|
          io << "kernel void " << msl_name << "("
          {% for input, i in inputs %}
            io << "device const float* s{{ i }} [[buffer({{ i }})]], "
          {% end %}
          io << "device float* dst [[buffer({{ inputs.size }})]], "
          {% for input, i in inputs %}
            io << "constant View& v{{ i }} [[buffer({{ inputs.size + 1 + i }})]], "
          {% end %}
          io << "constant uint& count [[buffer({{ 2 * inputs.size + 1 }})]], uint i [[thread_position_in_grid]]) {\n"
          io << "  if (i >= count) return;\n"
          {% for input, i in inputs %}
            io << "  float {{ input.id }} = s{{ i }}[offset_of(v{{ i }}, i)];\n"
          {% end %}
          {% for c, i in consts %}
            io << "  float c{{ i }} = " << ::Tensomite::Kernel.msl_float(C{{ i }}) << ";\n"
          {% end %}
          {% for s in statements %}
            {% operands = [s[2], s[3]].map { |v| v == "one" ? "1.0f" : (v == "two" ? "2.0f" : (v == "zero" ? "0.0f" : v)) } %}
            {% if s[1] == "neg" %}
              io << "  float {{ s[0].id }} = -{{ operands[0].id }};\n"
            {% elsif %w[tanh exp log sqrt].includes?(s[1]) %}
              io << "  float {{ s[0].id }} = precise::{{ s[1].id }}({{ operands[0].id }});\n"
            {% else %}
              io << "  float {{ s[0].id }} = {{ operands[0].id }} {{ s[1].id }} {{ operands[1].id }};\n"
            {% end %}
          {% end %}
          io << "  dst[i] = {{ (result == "zero" ? "0.0f" : result).id }};\n}\n"
        end
      end

      def cuda : String
        String.build do |io|
          io << "extern \"C\" __global__ void " << cuda_name << "("
          {% for input, i in inputs %}
            io << "const float* s{{ i }}, "
          {% end %}
          io << "float* dst, "
          {% for input, i in inputs %}
            io << "View v{{ i }}, "
          {% end %}
          io << "unsigned int count) {\n"
          io << "  unsigned int i = thread_index();\n"
          io << "  if (i >= count) return;\n"
          {% for input, i in inputs %}
            io << "  float {{ input.id }} = s{{ i }}[offset_of(v{{ i }}, i)];\n"
          {% end %}
          {% for c, i in consts %}
            io << "  float c{{ i }} = " << ::Tensomite::Kernel.msl_float(C{{ i }}) << ";\n"
          {% end %}
          {% for s in statements %}
            {% operands = [s[2], s[3]].map { |v| v == "one" ? "1.0f" : (v == "two" ? "2.0f" : (v == "zero" ? "0.0f" : v)) } %}
            {% if s[1] == "neg" %}
              io << "  float {{ s[0].id }} = -{{ operands[0].id }};\n"
            {% elsif %w[tanh exp log sqrt].includes?(s[1]) %}
              io << "  float {{ s[0].id }} = {{ s[1].id }}f({{ operands[0].id }});\n"
            {% else %}
              io << "  float {{ s[0].id }} = {{ operands[0].id }} {{ s[1].id }} {{ operands[1].id }};\n"
            {% end %}
          {% end %}
          io << "  dst[i] = {{ (result == "zero" ? "0.0f" : result).id }};\n}\n"
        end
      end

      def gradient(index : Int32) : ::Tensomite::Kernel?
        {% if gradients %}
          case index
          {% for g, i in gradients %}
            when {{ i }} then {{ g.id }}::INSTANCE
          {% end %}
          else raise ::Tensomite::Error.new("kernel #{name} has no input #{index}")
          end
        {% else %}
          nil
        {% end %}
      end

      # The kernel's single instance.
      INSTANCE = new
    end
  end
end
