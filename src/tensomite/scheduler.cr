module Tensomite
  # Turns a lazy graph into kernel runs.
  module Scheduler
    # Realises *root* and every unrealised node it depends on.
    def self.realize(root : Node, release : Bool = false) : Nil
      realize([root], release)
    end

    # Realises *roots* and every unrealised node they depend on, then waits once
    # for each of their devices.
    #
    # With *release*, each intermediate node computed here gives its buffer back
    # as soon as the last node here that reads it has run, so the memory does
    # not wait for the collector and later nodes can reuse it. Roots keep their
    # buffers, as do `Write` nodes and the buffers they write into; any other
    # node computed here can no longer be read. Devices run work in the order it
    # is submitted, so a buffer given back while a kernel that reads it is still
    # queued is only reused by work queued after it.
    #
    # 1. Walks sources depth first, with an explicit stack so deep graphs cannot
    #    exhaust the call stack, to list unrealised nodes in topological order.
    # 2. With *release*, counts each node's readers among those nodes.
    # 3. For each node, allocates its output buffer, or for `Write` takes its
    #    target's, and has its device execute it; with *release*, then counts
    #    down its sources' readers and releases those with none left.
    # 4. Waits for the roots' devices to finish.
    def self.realize(roots : Array(Node), release : Bool = false) : Nil
      roots = roots.reject(&.realized?)
      return if roots.empty?
      order = [] of Node
      seen = Set(Node).new
      roots.each { |root| visit(root, seen, order) }
      readers = release ? count_readers(order, roots) : nil
      order.each do |node|
        srcs = node.srcs
        buffer = node.op == Op::Write ? srcs.first.node.buffer! : node.device.alloc(node.bytesize)
        node.device.execute(node, buffer)
        node.realized(buffer)
        readers.try { |counts| release_read(srcs, counts) }
      end
      roots.map(&.device).uniq!(&.object_id).each(&.sync)
    ensure
      order.try(&.clear)
      seen.try(&.clear)
    end

    # Returns, for each node of *order* that may be released, the number of
    # times nodes of *order* read it. Roots, `Write` nodes and their targets are
    # left out, so they are never released.
    private def self.count_readers(order : Array(Node), roots : Array(Node)) : Hash(Node, Int32)
      counts = order.to_h { |node| {node, 0} }
      roots.each { |root| counts.delete(root) }
      order.each do |node|
        if node.op == Op::Write
          counts.delete(node)
          counts.delete(node.srcs.first.node)
        end
      end
      order.each do |node|
        node.srcs.each { |src| counts[src.node] += 1 if counts.has_key?(src.node) }
      end
      counts
    end

    # Counts down the readers of *srcs*, just read, and releases each with none left.
    private def self.release_read(srcs : Array(Tensor), counts : Hash(Node, Int32)) : Nil
      srcs.each do |src|
        node = src.node
        next unless count = counts[node]?
        if count <= 1
          counts.delete(node)
          node.release
        else
          counts[node] = count - 1
        end
      end
    end

    # Appends *root* and its unrealised dependencies to *order*, sources first. Each
    # node is pushed once to expand its sources, then again to be emitted. Raises
    # `Error` on reaching a released node, which can no longer be computed.
    private def self.visit(root : Node, seen : Set(Node), order : Array(Node)) : Nil
      stack = [{root, false}]
      while entry = stack.pop?
        node, expanded = entry
        if expanded
          order << node
        elsif !node.realized? && !seen.includes?(node)
          raise Error.new("node's buffer was released after the realize that computed it") if node.released?
          seen << node
          stack << {node, true}
          node.srcs.reverse_each { |src| stack << {src.node, false} }
        end
      end
    end
  end
end
