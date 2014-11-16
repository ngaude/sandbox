/*************************************************************************
 * Name:
 * Email:
 *
 * Compilation:  javac Point.java
 * Execution:
 * Dependencies: StdDraw.java
 *
 * Description: An immutable data type for points in the plane.
 *
 *************************************************************************/

import java.util.Comparator;

public class Point implements Comparable<Point> {

    // compare points by slope
    public final Comparator<Point> SLOPE_ORDER = new SlopeOrder();

    private final int x;                              // x coordinate
    private final int y;                              // y coordinate

    // create the point (x, y)
    public Point(int x, int y) {
        /* DO NOT MODIFY */
        this.x = x;
        this.y = y;
    }

    // plot this point to standard drawing
    public void draw() {
        /* DO NOT MODIFY */
        StdDraw.point(x, y);
    }

    // draw line between this point and that point to standard drawing
    public void drawTo(Point that) {
        /* DO NOT MODIFY */
        StdDraw.line(this.x, this.y, that.x, that.y);
    }

    // slope between this point and that point
    public double slopeTo(Point that) {
        if (null == that) throw new NullPointerException();
        
        double dx = that.x - x;
        double dy = that.y - y;
        double slope;
        
        
        if (dy == 0.0) {
            if (dx == 0.0) slope = Double.NEGATIVE_INFINITY;
            else slope = +0.0;
        }
        else if (dx == 0.0) slope = Double.POSITIVE_INFINITY;
        else slope = dy/dx;
        
        //StdOut.print(this.toString()+','+that+'='+slope+'\n');
        return slope;
    }

    // is this point lexicographically smaller than that one?
    // comparing y-coordinates and breaking ties by x-coordinates
    public int compareTo(Point that) {
        //StdOut.print(this.toString()+" ? "+that+'\n');
        if (y < that.y) return -1;
        else if (y > that.y) return +1;
        else {
            if (x < that.x) return -1;
            else if (x > that.x) return +1;
            else return 0;
        }
    }

    // return string representation of this point
    public String toString() {
        /* DO NOT MODIFY */
        return "(" + x + ", " + y + ")";
    }

    // unit test
    public static void main(String[] args) {
        /* YOUR CODE HERE */
    }
    
    private class SlopeOrder implements Comparator<Point>
    {
        public int compare(Point q1, Point q2)
        {
            double s1 = slopeTo(q1);
            double s2 = slopeTo(q2);
            
            if (s1 < s2) return -1;
            else if (s1 == s2) return 0;
            else return +1;
        
        }
    }
}
