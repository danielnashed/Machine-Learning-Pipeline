TraverseUp(Array *A, Node *n) {
  // Add n to A to maintain bottom up nature
  if (!n) return;
  A.add(n)

  // Go to parent
  Node *p = n.parent();
  if (!p) return;

  // For each child of p other than n, do a post order traversal
  foreach(Node *c in p.children) {
    if (c == n) continue;
    PostOrderTraversal(A, c);
  }
  // When done with adding all p's children, continue traversing up
  TraverseUp(A, p);
}

// Standard implementation of post order traversal
PostOrderTraversal(Array *A, Node *n) {
  if (!n) return;
  foreach(Node *c in n.children) {
    PostOrderTraversal(A, c);
    A.add(c);
  }
}