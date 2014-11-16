
public class Percolation {
    private int n;
    private WeightedQuickUnionUF uf;
    private boolean[] grid;
    private int top;
    private int bottom;
    public Percolation(int N) // create N-by-N grid, with all sites blocked
    {
        if (N <= 0) throw new IllegalArgumentException();

        n = N;
        grid = new boolean[N*N];
        uf = new WeightedQuickUnionUF(n*n+2);
        top = n*n;
        bottom = n*n + 1;
    }

    public void open(int i, int j) 
    // open site (row i, column j) if it is not already
    {
        if ((i < 1) || (j < 1) || (i > n) || (j > n)) 
            throw new IndexOutOfBoundsException();
        int x = (i-1)* n + (j-1);
        grid[x] = true;
        if ((i > 1) && (grid[x - n])) uf.union(x, x - n);
        if ((j > 1) && (grid[x - 1])) uf.union(x, x - 1);
        if ((i < n) && (grid[x + n])) uf.union(x, x + n);
        if ((j < n) && (grid[x + 1])) uf.union(x, x + 1);
        
        // if a top-row cell is opened then connect it to the top
        if (i == 1) uf.union(top, x);
        
        // if a bottom-raw cell is opened then connect it to the bottom
        if (i == n) uf.union(bottom, x);
    }

    public boolean isOpen(int i, int j) // is site (row i, column j) open?
    {
        if ((i < 1) || (j < 1) || (i > n) || (j > n)) 
            throw new IndexOutOfBoundsException();
        int x = (i-1)* n + (j-1);
        return grid[x];
    }

    public boolean isFull(int i, int j) // is site (row i, column j) full?
    {
        if ((i < 1) || (j < 1) || (i > n) || (j > n)) 
            throw new IndexOutOfBoundsException();
        int x = (i-1)* n + (j-1);
        return grid[x] && uf.connected(x, top);
        
    }

    public boolean percolates()              // does the system percolate?
    {
        return uf.connected(top, bottom); 
    }

    /*
    private String toText() {
        String txt = "";
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                if (isFull(i, j)) txt += '~';
                else if (isOpen(i, j)) txt += ' ';
                else txt += '#';
            }
            txt += '\n';
        }
        return txt;
    }
    */

    public static void main(String[] args) {
        Percolation p = new Percolation(2);
        System.out.println(p);
        System.out.println(p.percolates());
        p.open(1, 1);
        p.open(2, 2);
        p.open(1, 2);
        System.out.println(p);
        System.out.println(p.percolates());

    }

}

