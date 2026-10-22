module Tensomite
  # Turns a lazy graph into kernel runs.
  module Scheduler
    # Realises *root* and every unrealised node it depends on.
    #
    # 1. Walks sources depth first to list unrealised nodes in topological order.
    # 2. For each node, allocates its output buffer and has its device execute it.
    # 3. Waits for the root's device to finish.
    def self.realize(root : Node) : Nil
      return if root.realized?
      order = [] of Node
      visit(root, Set(Node).new, order)
      order.each do |node|
        buffer = node.device.alloc(node.bytesize)
        node.device.execute(node, buffer)
        node.realized(buffer)
      end
      root.device.sync
    end

    private def self.visit(node : Node, seen : Set(Node), order : Array(Node)) : Nil
      return if node.realized? || seen.includes?(node)
      seen << node
      node.srcs.each { |src| visit(src.node, seen, order) }
      order << node
    end
  end
end
