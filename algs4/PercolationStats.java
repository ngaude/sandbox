import java.util.InputMismatchException;

public class PercolationStats 
{
    private double[] pthreshold;
    private int n, t;

    public PercolationStats(int N, int T)    
    // perform T independent computational experiments on an N-by-N grid
    {
        if ((N <= 0) || (T <= 0)) throw new IllegalArgumentException();
        n = N;
        t = T;

        pthreshold = new double[t];

        for (int e = 0; e < T; e++) {
            Percolation p = new Percolation(N);
            int opened = 0;
            while (!p.percolates()) {
                int row, col;
                do {
                    row = 1 + StdRandom.uniform(N);
                    col = 1 + StdRandom.uniform(N);
                } while (p.isOpen(row, col));
                p.open(row, col);
                opened++;
            }
            pthreshold[e] = (opened*1.0)/(n*n);
        }
    }

    public double mean()                     
    // sample mean of percolation threshold
    {
        return StdStats.mean(pthreshold);
    }

    public double stddev()                   
    // sample standard deviation of percolation threshold
    {
        return StdStats.stddev(pthreshold);
    }

    public double confidenceLo()             
    // returns lower bound of the 95% confidence interval
    {
        return mean() - 1.96*stddev()/Math.sqrt(t);
    }
    public double confidenceHi()             
    // returns upper bound of the 95% confidence interval
    {
        return mean() + 1.96*stddev()/Math.sqrt(t);
    }

    public static void main(String[] args)   
    // test client, described below
    {
        int N, T;
        try {
            N = StdIn.readInt();
            T = StdIn.readInt();
        } catch (InputMismatchException e) {
            N = 8;
            T = 8;
        }
        PercolationStats ps = new PercolationStats(N, T);
        System.out.println("mean                    = " + ps.mean());
        System.out.println("stddev                  = " + ps.stddev());
        System.out.println("95% confidence interval = " 
                + ps.confidenceLo() + ", " + ps.confidenceHi());
    }
}