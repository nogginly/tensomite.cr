module Tensomite
  # Turns a lazy graph into kernel runs.
  module Scheduler
    # Realises *root* and every unrealised node it depends on.
    def self.realize(root : Node) : Nil
      realize([root])
    end

    # Realises *roots* and every unrealised node they depend on, then waits once
    # for each of their devices.
    #
    # 1. Walks sources depth first, with an explicit stack so deep graphs cannot
    #    exhaust the call stack, to list unrealised nodes in topological order.
    # 2. For each node, allocates its output buffer, or for `Write` takes its
    #    target's, and has its device execute it.
    # 3. Waits for the roots' devices to finish.
    def self.realize(roots : Array(Node)) : Nil
      roots = roots.reject(&.realized?)
      return if roots.empty?
      order = [] of Node
      seen = Set(Node).new
      roots.each { |root| visit(root, seen, order) }
      order.each do |node|
        buffer = node.op == Op::Write ? node.srcs.first.node.buffer! : node.device.alloc(node.bytesize)
        node.device.execute(node, buffer)
        node.realized(buffer)
      end
      roots.map(&.device).uniq!(&.object_id).each(&.sync)
    end

    # Appends *root* and its unrealised dependencies to *order*, sources first. Each
    # node is pushed once to expand its sources, then again to be emitted.
    private def self.visit(root : Node, seen : Set(Node), order : Array(Node)) : Nil
      stack = [{root, false}]
      while entry = stack.pop?
        node, expanded = entry
        if expanded
          order << node
        elsif !node.realized? && !seen.includes?(node)
          seen << node
          stack << {node, true}
          node.srcs.reverse_each { |src| stack << {src.node, false} }
        end
      end
    end
  end
end
