# encoding: utf-8
"""
Một triển khai của thuật toán FP-growth bằng Python.
"""

from collections import defaultdict, namedtuple

# Thông tin tác giả gốc, phiên bản này được cập nhật bởi lina.
__author__ = 'Eric Naeseth <eric@naeseth.com>'
__copyright__ = 'Bản quyền © 2009 Eric Naeseth'
__license__ = 'Giấy phép MIT'

def find_frequent_itemsets(transactions, minimum_support, include_support=False):
    """
    Tìm các tập mục thường xuyên trong các giao dịch cho trước bằng cách sử dụng FP-growth.
    Hàm này trả về một generator thay vì một danh sách các mục được tạo trước.
    """
    items = defaultdict(lambda: 0)

    # Tải các giao dịch đã cho và đếm số lần xuất hiện của các mục riêng lẻ.
    for transaction in transactions:
        for item in transaction:
            items[item] += 1

    # Loại bỏ các mục không thường xuyên khỏi từ điển hỗ trợ mục.
    items = dict((item, support) for item, support in items.items()
        if support >= minimum_support)

    # Xây dựng cây FP. Trước khi thêm bất kỳ giao dịch nào vào cây,
    # chúng phải được loại bỏ các mục không thường xuyên và các mục còn lại phải được sắp xếp giảm dần theo tần suất.
    def clean_transaction(transaction):
        transaction = filter(lambda v: v in items, transaction)
        transaction_list = list(transaction)
        transaction_list.sort(key=lambda v: items[v], reverse=True)
        return transaction_list

    master = FPTree()
    for transaction in map(clean_transaction, transactions):
        master.add(transaction)

    def find_with_suffix(tree, suffix):
        for item, nodes in tree.items():
            support = sum(n.count for n in nodes)
            if support >= minimum_support and item not in suffix:
                found_set = [item] + suffix
                yield (found_set, support) if include_support else found_set

                cond_tree = conditional_tree_from_paths(tree.prefix_paths(item))
                for s in find_with_suffix(cond_tree, found_set):
                    yield s

    for itemset in find_with_suffix(master, []):
        yield itemset

class FPTree(object):
    """
    Cây FP.
    """

    Route = namedtuple('Route', 'head tail')

    def __init__(self):
        self._root = FPNode(self, None, None)
        self._routes = {}

    @property
    def root(self):
        return self._root

    def add(self, transaction):
        point = self._root
        for item in transaction:
            next_point = point.search(item)
            if next_point:
                next_point.increment()
            else:
                next_point = FPNode(self, item)
                point.add(next_point)
                self._update_route(next_point)
            point = next_point

    def _update_route(self, point):
        assert self is point.tree

        try:
            route = self._routes[point.item]
            route[1].neighbor = point
            self._routes[point.item] = self.Route(route[0], point)
        except KeyError:
            self._routes[point.item] = self.Route(point, point)

    def items(self):
        for item in self._routes.keys():
            yield (item, self.nodes(item))

    def nodes(self, item):
        try:
            node = self._routes[item][0]
        except KeyError:
            return

        while node:
            yield node
            node = node.neighbor

    def prefix_paths(self, item):
        def collect_path(node):
            path = []
            while node and not node.root:
                path.append(node)
                node = node.parent
            path.reverse()
            return path

        return (collect_path(node) for node in self.nodes(item))

    def inspect(self):
        print('Tree:')
        self.root.inspect(1)

        print('Routes:')
        for item, nodes in self.items():
            print('  %r' % item)
            for node in nodes:
                print('    %r' % node)

def conditional_tree_from_paths(paths):
    tree = FPTree()
    condition_item = None
    items = set()

    for path in paths:
        if condition_item is None:
            condition_item = path[-1].item

        point = tree.root
        for node in path:
            next_point = point.search(node.item)
            if not next_point:
                items.add(node.item)
                count = node.count if node.item == condition_item else 0
                next_point = FPNode(tree, node.item, count)
                point.add(next_point)
                tree._update_route(next_point)
            point = next_point

    assert condition_item is not None

    for path in tree.prefix_paths(condition_item):
        count = path[-1].count
        for node in reversed(path[:-1]):
            node._count += count

    return tree

class FPNode(object):
    """Một nút trong cây FP."""

    def __init__(self, tree, item, count=1):
        self._tree = tree
        self._item = item
        self._count = count
        self._parent = None
        self._children = {}
        self._neighbor = None

    def add(self, child):
        if not isinstance(child, FPNode):
            raise TypeError("Chỉ có thể thêm các FPNode khác làm con")

        if not child.item in self._children:
            self._children[child.item] = child
            child.parent = self

    def search(self, item):
        try:
            return self._children[item]
        except KeyError:
            return None

    def __contains__(self, item):
        return item in self._children

    @property
    def tree(self):
        return self._tree

    @property
    def item(self):
        return self._item

    @property
    def count(self):
        return self._count

    def increment(self):
        if self._count is None:
            raise ValueError("Nút gốc không có số đếm liên quan.")
        self._count += 1

    @property
    def root(self):
        return self._item is None and self._count is None

    @property
    def leaf(self):
        return len(self._children) == 0

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("Một nút phải có một FPNode làm cha mẹ.")
        if value and value.tree is not self.tree:
            raise ValueError("Không thể có cha mẹ từ cây khác.")
        self._parent = value

    @property
    def neighbor(self):
        return self._neighbor

    @neighbor.setter
    def neighbor(self, value):
        if value is not None and not isinstance(value, FPNode):
            raise TypeError("Một nút phải có một FPNode làm hàng xóm.")
        if value and value.tree is not self.tree:
            raise ValueError("Không thể có hàng xóm từ cây khác.")
        self._neighbor = value

    @property
    def children(self):
        return tuple(self._children.values())

    def inspect(self, depth=0):
        print(('  ' * depth) + repr(self))
        for child in self.children:
            child.inspect(depth + 1)

    def __repr__(self):
        if self.root:
            return "<%s (gốc)>" % type(self).__name__
        return "<%s %r (%r)>" % (type(self).__name__, self.item, self.count)

if __name__ == '__main__':
    from optparse import OptionParser
    import csv

    p = OptionParser(usage='%prog data_file')
    p.add_option('-s', '--minimum-support', dest='minsup', type='int',
        help='Hỗ trợ tối thiểu cho tập mục (mặc định: 2)')
    p.add_option('-n', '--numeric', dest='numeric', action='store_true',
        help='Chuyển đổi các giá trị trong tập dữ liệu thành số (mặc định: false)')
    p.set_defaults(minsup=2)
    p.set_defaults(numeric=False)

    options, args = p.parse_args()
    if len(args) != 1:
        'C:/Users/TRITON/.spyder-py3/baitap.txt'
    transactions = []
    with open(args[0]) as database:
        for row in csv.reader(database):
            if options.numeric:
                transaction = []
                for item in row:
                    transaction.append(int(item))
                transactions.append(transaction)
            else:
                transactions.append(row)

    result = []
    for itemset, support in find_frequent_itemsets(transactions, options.minsup, True):
        result.append((itemset, support))

    result = sorted(result, key=lambda i: i[0])
    for itemset, support in result:
        print(str(itemset) + ' ' + str(support))


p.add_option('-t', '--theme', dest='theme', type='str',
    help='Chọn theme cho giao diện đầu ra (mặc định: default)')
p.set_defaults(theme='default')
