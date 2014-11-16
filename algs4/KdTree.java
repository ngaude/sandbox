public class KdTree {
    private SET<Point2D> pp;
    // construct an empty set of points

    public KdTree() {
        this.pp = new SET<Point2D>();
    }

    // is the set empty?
    public boolean isEmpty() {
        return pp.isEmpty();
    }

    // number of points in the set 
    public int size() {
        return pp.size();
    }

    // add the point to the set (if it is not already in the set)
    public void insert(Point2D p) {
        pp.add(p);
    }

    // does the set contain point p?
    public boolean contains(Point2D p) {
        return pp.contains(p);
    }

    // draw all points to standard draw
    public void draw() {
        for (Point2D p : pp) 
            p.draw();
    }

    // all points that are inside the rectangle
    public Iterable<Point2D> range(RectHV rect) {
        ResizingArrayStack<Point2D> it = new ResizingArrayStack<Point2D>();
        for (Point2D p : pp) { 
            if (rect.contains(p)) {
                it.push(p);
            }
        }
        return it;
    }
    // a nearest neighbor in the set to point p;
    // null if the set is empty
    public Point2D nearest(Point2D p) {
        double distance = Double.MAX_VALUE;
        Point2D nearest = null;
        for (Point2D other : pp) {
            if (p.distanceTo(other) < distance) {
                distance = p.distanceTo(other);
                nearest = other;
            }
        }
        return nearest;
    }
}

