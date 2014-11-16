public class Brute {
    
    private static Point[] q;

    public static void main(String[] args) {
        
        String filename = args[0];
        In in = new In(filename);
        int N = in.readInt();
        q = new Point[N];
        StdDraw.setXscale(0, 32768);
        StdDraw.setYscale(0, 32768);
        
        for (int i = 0; i < N; i++) {
            int x = in.readInt();
            int y = in.readInt();
            q[i] = new Point(x, y);
            q[i].draw();
        }

        colinear();
    }
    
    private static void colinear() {
        int i, j, k, l;
        for (i = 0;  i < q.length; i++) {
            for (j = i+1;  j < q.length; j++) {
                double a = q[i].slopeTo(q[j]);
                for (k = j+1;  k < q.length; k++) {
                    double b = q[i].slopeTo(q[k]);
                    for (l = k+1;  l < q.length; l++) {
                        double c = q[i].slopeTo(q[l]);
                        if ((a == b) && (b == c)) {

                            /* order q[i],q[j],q[k],q[l] */
                            Point [] seg = new Point[]{q[i], q[j], q[k], q[l]};
                            Selection.sort(seg);
                            StdOut.print(seg[0] + " -> " + seg[1] 
                                    + " -> " + seg[2] + " -> " + seg[3] + '\n');
                            seg[0].drawTo(seg[3]);
                        }
                    }
                }
            }
        }
    }
}