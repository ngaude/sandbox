import java.util.Comparator;

public class Fast {

    private static MyPoint[] q;
    private static boolean[][] b;

    private class MyPoint extends Point {
        private int id;
        private MyPoint(int x, int y, int pid) {
            super(x, y);
            id = pid;
        }
        //
        //        public String toString() {
        //            /* DO NOT MODIFY */
        //            return super.toString() + "[" + id + "]";
        //        }

    }

    private Fast() { }

    public static void main(String[] args) {

        String filename = args[0];
        In in = new In(filename);
        int N = in.readInt();

        StdDraw.setXscale(0, 32768);
        StdDraw.setYscale(0, 32768);

        q = new MyPoint[N];
        b = new boolean[N][N];

        Fast enclosingInstance = new Fast();

        for (int i = 0; i < N; i++) {
            int x = in.readInt();
            int y = in.readInt();
            q[i] = enclosingInstance.new MyPoint(x, y, i);
            q[i].draw();
        }

        colinear();
    }

    private static void colinear() {

        for (int i = 0;  i < q.length - 1; i++) {

            InnerQuick3way.sort(q, i + 1, q.length - 1, q[i].SLOPE_ORDER);

            double lastSlope = Double.NEGATIVE_INFINITY;
            double currSlope;
            int lastId = i;
            int currId = i; 

            while (currId++ < q.length) {
                if (currId == q.length) {
                    currSlope = Double.NEGATIVE_INFINITY;
                }
                else currSlope = q[i].slopeTo(q[currId]);

                if ((currSlope != Double.NEGATIVE_INFINITY) 
                        || (currId == q.length)) {
                    if (currSlope != lastSlope) {
                        if (currId-lastId >= 3) {
                            Point[] seg = new Point[1+currId-lastId];
                            seg[0] = q[i];
                            for (int j = lastId; j < currId; j++) {
                                seg[j+1-lastId] = q[j]; 
                            }

                            /* check if one of these segment 
                             * has already been printed ? */
                            MyPoint orig, dest;
                            orig = (MyPoint) seg[0];
                            dest = (MyPoint) seg[1];
                            if (!b[orig.id][dest.id]) {

                                /* Quick3way ? or selection ? */
                                Quick3way.sort(seg);
                                String s = seg[0].toString();
                                for (int j = 1; j < seg.length; j++) {
                                    s += " -> " + seg[j];
                                }
                                
                                StdOut.println(s);
                                seg[0].drawTo(seg[seg.length-1]);

                                /* once printed out, eliminate all sub segment */
                                for (int j = 0; j < seg.length; j++) {
                                    for (int k = 0; k < seg.length; k++) {
                                        orig = (MyPoint) seg[j];
                                        dest = (MyPoint) seg[k];

                                        b[orig.id][dest.id] = true;
                                    }
                                }
                            }

                        }
                        lastId = currId;
                        lastSlope = currSlope;
                    }
                }
            }
        }
    }

    private static class InnerQuick3way {
        
        private static Comparator<Point> comp;
        
        // This class should not be instantiated.
        private InnerQuick3way() { }

        private static void shuffle(Point[] a, int lo, int hi) {
            for (int i = lo; i <= hi; i++) {
                int r = i + StdRandom.uniform(hi+1-i);     // between i and hi
                Point temp = a[i];
                a[i] = a[r];
                a[r] = temp;
            }
        }

        /**
         * Rearranges the array in ascending order, 
         * using a comparator specific order.
         * @param a the array to be sorted
         */
        public static void sort(Point[] a, int lo, int hi, Comparator<Point> p) {
            comp = p;
            shuffle(a, lo, hi);
            sort(a, lo, hi);

            assert isSorted(a, lo, hi);
        }

        // quicksort the subarray a[lo .. hi] using 3-way partitioning
        private static void sort(Point[] a, int lo, int hi) { 
            if (hi <= lo) return;
            int lt = lo, gt = hi;
            Point v = a[lo];
            int i = lo;
            while (i <= gt) {
                //int cmp = a[i].compareTo(v);
                int cmp = comp.compare(a[i], v);

                if      (cmp < 0) exch(a, lt++, i++);
                else if (cmp > 0) exch(a, i, gt--);
                else              i++;
            }

            // a[lo..lt-1] < v = a[lt..gt] < a[gt+1..hi]. 
            sort(a, lo, lt-1);
            sort(a, gt+1, hi);
            assert isSorted(a, lo, hi);
        }



        /***********************************************************************
         *  Helper sorting functions
         ***********************************************************************/

        // is v < w ?
        private static boolean less(Point v, Point w) {
            return (comp.compare(v, w) < 0);
        }

        // exchange a[i] and a[j]
        private static void exch(Object[] a, int i, int j) {
            Object swap = a[i];
            a[i] = a[j];
            a[j] = swap;
        }


        /***********************************************************************
         *  Check if array is sorted - useful for debugging
         ***********************************************************************/

        private static boolean isSorted(Point[] a, int lo, int hi) {
            for (int i = lo + 1; i <= hi; i++)
                if (less(a[i], a[i-1])) return false;
            return true;
        }
    }

}
