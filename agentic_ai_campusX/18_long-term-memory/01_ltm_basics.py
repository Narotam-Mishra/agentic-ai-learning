
# long term memory basic example

from langgraph.store.memory import InMemoryStore

# create store
store = InMemoryStore()

# create a namespace
namespace = ("user", "u1")

# creating memoriesß

# adding memories
store.put(namespace, "1", {"data": "User likes pizza"})
store.put(namespace, "2", {"data": "User prefers dark mode"})

# another namespace
namespace2 = ("user", "u2")

# add memories
store.put(namespace2, "1", {"data": "User likes pasta"})
store.put(namespace2, "2", {"data": "User prefers grid style navigation"})

# retrieving memories
val = store.get(namespace, "1")
print(f"value: {val}")

# retrieving all memories
items = store.search(namespace2)

for item in items:
    print(f"Memory content: {item.value}")