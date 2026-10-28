module Tensomite
  # Turns a lazy graph into kernel runs.
  module Scheduler
    # Realises *root* and every unrealised node it depends on.
    #
    # 1. Walks sources depth first, with an explicit stack so deep graphs cannot
    #    exhaust the call stack, to list unrealised nodes in topological order.
    # 2. For each node, allocates its output buffer, or for `Write` takes its
    #    target's, and has its device execute it.
    # 3. Waits for the root's device to finish.
    def self.realize(root : Node) : Nil
      return if root.realized?
      order = [] of Node
      visit(root, Set(Node).new, order)
      order.each do |node|
        buffer = node.op == Op::Write ? node.srcs.first.node.buffer! : node.device.alloc(node.bytesize)
        node.device.execute(node, buffer)
        node.realized(buffer)
      end
      root.device.sync
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
