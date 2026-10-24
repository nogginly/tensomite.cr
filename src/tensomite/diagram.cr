module Tensomite
  struct Tensor
    # Returns a Mermaid flowchart of the graph behind this tensor, sources at the top.
    #
    # With *depth* `nil`, every node is a box: its op (or `leaf`), its name if it
    # has one, and its shape, with the dtype when it is not `F32`. A source read
    # through a view other than the node itself labels its edge with the kind of
    # view (`reshape`, `slice`, `expand` or `permute`) and the shape it reads.
    # Leaves have a green border and this tensor's node a red one. Of a graph
    # larger than *max_nodes*, the nodes nearest this one are drawn and the rest
    # stand behind a single box.
    #
    # With *depth* `n`, nodes are grouped by their `Tensomite.scope` path cut to
    # `n` levels, and each group is one box, with its op count and the number of
    # elements of *params* it reads first, per instance when it is folded. Edges between groups carry the shape
    # that crosses. Sibling scopes of one name whose graphs match are folded into
    # one, marked `×k`, so a model's repeated blocks show once. Scopes above the
    # last level are drawn as subgraphs. Leaves outside *params* are inputs, each
    # in a box of its own wherever it was made, and scalars are left out.
    #
    # Box ids follow the graph's order, so the same graph gives the same text.
    #
    # ```
    # w = Tensor.from(f32(1, 2, 3, 4, 5, 6), [2, 3]).named("w")
    # puts (w * w).sum.to_mermaid
    # ```
    def to_mermaid(max_nodes : Int32 = 200, depth : Int32? = nil, params : Enumerable(Tensor) = [] of Tensor) : String
      String.build { |io| to_mermaid(io, max_nodes, depth, params) }
    end

    # Writes the flowchart `#to_mermaid` returns to *io*.
    def to_mermaid(io : IO, max_nodes : Int32 = 200, depth : Int32? = nil, params : Enumerable(Tensor) = [] of Tensor) : Nil
      if depth
        raise ArgumentError.new("depth must be at least 1, not #{depth}") unless depth >= 1
        Diagram::Summary.new(node, depth, params.map(&.node).to_set).write(io)
      else
        raise ArgumentError.new("max_nodes must be at least 1, not #{max_nodes}") unless max_nodes >= 1
        Diagram.ops(io, node, max_nodes)
      end
    end
  end

  # Mermaid flowcharts of graphs; see `Tensor#to_mermaid`.
  module Diagram
    LEAF = "stroke:#2b8a3e,stroke-width:2px"
    ROOT = "stroke:#c92a2a,stroke-width:2px"

    # Returns the nodes *root* depends on, itself included, sources before the
    # nodes that use them.
    def self.graph_order(root : Node) : Array(Node)
      order = [] of Node
      seen = Set(Node).new
      stack = [{root, false}]
      while entry = stack.pop?
        node, expanded = entry
        if expanded
          order << node
        elsif seen.add?(node)
          stack << {node, true}
          node.srcs.reverse_each { |src| stack << {src.node, false} }
        end
      end
      order
    end

    # Writes the flowchart of every node, or of the *max_nodes* nearest *root*.
    #
    # 1. Lists the graph in order, and when it is too large, keeps the nodes a
    #    breadth-first walk from *root* reaches first.
    # 2. Writes a box per kept node, numbered in graph order, and one for the rest.
    # 3. Writes an edge per source, from the hidden box when the source is hidden,
    #    and once per node from it.
    # 4. Styles leaves and the root.
    def self.ops(io : IO, root : Node, max_nodes : Int32) : Nil
      order = graph_order(root)
      kept = order.size > max_nodes ? nearest(root, max_nodes) : order.to_set
      ids = {} of Node => String
      order.each { |node| ids[node] = "n#{ids.size}" if kept.includes?(node) }
      hidden = order.size - kept.size
      io << "flowchart TB\n"
      ids.each { |node, id| io << "    " << id << "[\"" << escape(label(node)) << "\"]\n" }
      io << "    more[\"… " << hidden << " more nodes\"]\n" if hidden > 0
      ids.each do |node, id|
        from_hidden = false
        node.srcs.each do |src|
          if source = ids[src.node]?
            edge(io, source, id, view_label(src))
          elsif !from_hidden
            edge(io, "more", id, nil)
            from_hidden = true
          end
        end
      end
      leaves = ids.compact_map { |node, id| id if node.op.nil? && node != root }
      style(io, "leaf", LEAF, leaves)
      style(io, "root", ROOT, [ids[root]])
    end

    # Returns the first *count* nodes a breadth-first walk from *root* reaches.
    private def self.nearest(root : Node, count : Int32) : Set(Node)
      kept = Set{root}
      queue = Deque{root}
      while kept.size < count && (node = queue.shift?)
        node.srcs.each do |src|
          break if kept.size >= count
          queue << src.node if kept.add?(src.node)
        end
      end
      kept
    end

    # Returns a node's box text: name, op and shape, and dtype when not `F32`.
    def self.label(node : Node) : String
      String.build do |io|
        io << node.name << " · " if node.name
        if op = node.op
          io << op
          io << ' ' << node.kernel.try(&.name) if op.kernel?
          io << " axis " << node.axis if node.axis
        else
          io << "leaf"
        end
        io << ' ' << shape(node.shape)
        io << ' ' << node.dtype unless node.dtype.f32?
      end
    end

    # Returns the kind and shape of the view *src* reads its node through, or `nil`
    # when it reads the node as it is.
    def self.view_label(src : Tensor) : String?
      view, node = src.view, src.node
      return if view.contiguous? && view.shape == node.shape
      kind = if view.strides.includes?(0) && view.size > 1
               "expand"
             elsif view.size < node.size
               "slice"
             elsif view.contiguous?
               "reshape"
             else
               "permute"
             end
      "#{kind} #{shape(view.shape)}"
    end

    def self.shape(dims : Array(Int32)) : String
      "[#{dims.join(',')}]"
    end

    # Returns *count* to three significant figures, with `k` or `M`.
    def self.count(count : Int) : String
      return count.to_s if count < 1000
      value, suffix = count < 1_000_000 ? {count / 1e3, "k"} : {count / 1e6, "M"}
      digits = value >= 100 ? 0 : (value >= 10 ? 1 : 2)
      "#{value.round(digits).to_s.rchop(".0")}#{suffix}"
    end

    def self.escape(text : String) : String
      text.gsub('"', "#quot;")
    end

    def self.edge(io : IO, from : String, to : String, label : String?) : Nil
      io << "    " << from
      io << " -- \"" << escape(label) << '"' if label
      io << " --> " << to << '\n'
    end

    def self.style(io : IO, name : String, style : String, ids : Array(String)) : Nil
      return if ids.empty?
      io << "    classDef " << name << ' ' << style << '\n'
      io << "    class " << ids.join(',') << ' ' << name << '\n'
    end

    # The grouped flowchart of `Tensor#to_mermaid` with a depth.
    class Summary
      # A box: the nodes of one scope cut to the depth, or one unscoped node.
      private class Group
        getter id : String
        getter scope : Scope?
        getter nodes = [] of Node
        property params = 0_i64

        def initialize(@id, @scope)
        end
      end

      @order : Array(Node)
      # Each scope's children in the order the graph first meets them.
      @children = {} of Scope? => Array(Scope)
      @canonical = {} of Scope => Scope
      @folds = {} of Scope => Int32
      @groups = {} of String => Group
      @group_of = {} of Node => Group
      @subgraphs = 0

      def initialize(@root : Node, @depth : Int32, @params : Set(Node))
        @order = Diagram.graph_order(@root)
        @order.each { |node| register(node.scope) }
      end

      # Writes the flowchart.
      #
      # 1. Assigns each node to a group: its scope cut to the depth and mapped onto
      #    the first of a folded family, or a group of its own when unscoped.
      #    Params are counted towards the group of their first reader; scalars
      #    and params get no box.
      # 2. Writes the groups, nested in subgraphs for their outer scopes.
      # 3. Writes one edge per pair of groups that a source joins, labelled with
      #    the shape of the first such source.
      def write(io : IO) : Nil
        @order.each { |node| assign(node) }
        io << "flowchart TB\n"
        write_scope(io, nil, 1)
        written = Set({String, String}).new
        @order.each do |node|
          next unless to = @group_of[node]?
          node.srcs.each do |src|
            next unless from = @group_of[src.node]?
            next_instance = next_instance?(src.node, node)
            next if from == to && !next_instance
            next unless written.add?({from.id, to.id})
            if next_instance
              io << "    " << from.id << " -. \"next\" .-> " << to.id << '\n'
            else
              Diagram.edge(io, from.id, to.id, Diagram.shape(src.shape))
            end
          end
        end
        inputs = @groups.values.compact_map { |group| group.id if group.nodes.all?(&.op.nil?) }
        Diagram.style(io, "leaf", LEAF, inputs)
        Diagram.style(io, "root", ROOT, @group_of[@root]?.try { |group| [group.id] } || [] of String)
      end

      # Returns true when *from* and *to* lie in different members of one folded
      # family, as when one block's output feeds the next block.
      private def next_instance?(from : Node, to : Node) : Bool
        a = from.scope.try(&.path) || return false
        b = to.scope.try(&.path) || return false
        a.zip?(b) do |outer, inner|
          next if outer.same?(inner)
          return !inner.nil? && outer.parent == inner.parent && outer.name == inner.name && folded?(family(outer))
        end
        false
      end

      private def register(scope : Scope?) : Nil
        return unless scope
        register(scope.parent)
        siblings = @children[scope.parent] ||= [] of Scope
        siblings << scope unless siblings.includes?(scope)
      end

      private def assign(node : Node) : Nil
        if node.op.nil?
          return if node.size == 1 && node.name.nil?
          if @params.includes?(node)
            reader = @order.find { |other| other.srcs.any?(&.node.same?(node)) }
            if reader && (group = group_for(reader))
              group.params += node.size
            end
            return
          end
        end
        group_for(node).try do |group|
          group.nodes << node unless group.nodes.includes?(node)
          @group_of[node] = group
        end
      end

      private def group_for(node : Node) : Group?
        scope = node.op ? node.scope.try { |inner| canonical(cut(inner)) } : nil
        key = scope ? "s#{scope.object_id}" : "n#{node.object_id}"
        @groups[key] ||= Group.new("b#{@groups.size}", scope)
      end

      private def cut(scope : Scope) : Scope
        path = scope.path
        path[Math.min(@depth, path.size) - 1]
      end

      # Returns the scope that stands for *scope* once families of matching
      # siblings are folded: its counterpart, by name and position, inside the
      # first of each folded family on its path.
      private def canonical(scope : Scope) : Scope
        @canonical[scope] ||= begin
          family = family(scope)
          base = folded?(family) ? family.first : scope
          parent = scope.parent.try { |outer| canonical(outer) }
          if parent == base.parent
            base
          else
            position = family.index!(scope)
            (@children[parent]? || [] of Scope).select(&.name.==(scope.name))[position]? || base
          end
        end
      end

      private def family(scope : Scope) : Array(Scope)
        (@children[scope.parent]? || [scope]).select(&.name.==(scope.name))
      end

      private def folded?(family : Array(Scope)) : Bool
        return false if family.size < 2
        @folds[family.first] ||= family.map { |scope| signature(scope) }.uniq!.size == 1 ? family.size : 1
        @folds[family.first] > 1
      end

      # Returns a description of the nodes inside *scope*: each one's op, kernel,
      # shape and scope names below *scope*, in graph order, and the params they read.
      private def signature(scope : Scope) : String
        String.build do |io|
          @order.each do |node|
            next unless (path = node.scope.try(&.path)) && (at = path.index(scope))
            io << node.op << node.kernel.try(&.name) << node.shape
            path[(at + 1)..].each { |inner| io << '/' << inner.name }
            node.srcs.each { |src| io << " p" << src.node.size if @params.includes?(src.node) }
            io << ';'
          end
        end
      end

      # Writes the groups of the scopes inside *parent*, at *level*, and the
      # unscoped groups when *parent* is `nil`.
      private def write_scope(io : IO, parent : Scope?, level : Int32) : Nil
        indent = "    " * level
        if parent.nil?
          @groups.each_value { |group| write_group(io, group, indent, nil) if group.scope.nil? }
        end
        (@children[parent]? || [] of Scope).each do |scope|
          next unless canonical(scope).same?(scope)
          own = @groups["s#{scope.object_id}"]?
          inner = level < @depth && contains_groups?(scope)
          if inner
            io << indent << "subgraph sub" << (@subgraphs += 1) << " [\"" << Diagram.escape(title(scope)) << "\"]\n"
            write_group(io, own, indent + "    ", "own ops") if own
            write_scope(io, scope, level + 1)
            io << indent << "end\n"
          elsif own
            write_group(io, own, indent, title(scope))
          end
        end
      end

      private def contains_groups?(scope : Scope) : Bool
        (@children[scope]? || [] of Scope).any? do |child|
          @groups.has_key?("s#{child.object_id}") || contains_groups?(child)
        end
      end

      private def write_group(io : IO, group : Group, indent : String, title : String?) : Nil
        text = if title
                 copies = copies(group.scope)
                 ops = group.nodes.size // copies
                 detail = [ops == 1 ? "1 op" : "#{ops} ops"]
                 detail << "#{Diagram.count(group.params // copies)} params" if group.params > 0
                 "#{title}<br/>#{detail.join(" · ")}#{" each" if copies > 1}"
               else
                 Diagram.label(group.nodes.first)
               end
        io << indent << group.id << "[\"" << Diagram.escape(text) << "\"]\n"
      end

      # Returns how many instances a group of *scope* stands for: the product of
      # the sizes of the folded families on its path.
      private def copies(scope : Scope?) : Int32
        return 1 unless scope
        scope.path.product { |outer| folded?(family(outer)) ? family(outer).size : 1 }
      end

      private def title(scope : Scope) : String
        family = family(scope)
        folds = folded?(family) ? family.size : 1
        folds > 1 ? "#{scope.name} ×#{folds}" : scope.name
      end
    end
  end
end
