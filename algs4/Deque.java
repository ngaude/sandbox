import java.util.Iterator;
import java.util.NoSuchElementException;

public class Deque<Item> implements Iterable<Item> {

    private Item[] q;
    private int N = 0;
    private int first = 0;
    private int last = 0;
    
    public Deque()                           
    // construct an empty deque
    {
        q = (Item[]) new Object[2];
    }
    
    public boolean isEmpty()                 
    // is the deque empty?
    {
        return (N == 0);
    }
    public int size()                        
    // return the number of items on the deque
    {
        return N;
    }
    public void addFirst(Item item)          
    // insert the item at the front
    {
        if (item == null) throw new NullPointerException();
        
        if (N == q.length) resize(q.length*2);
        
        if (!isEmpty()) first--;
        if (first == -1) first = q.length-1;
        
        q[first] = item;
        N++;
    }
    
    public void addLast(Item item)           
    // insert the item at the end
    {
        if (item == null) throw new NullPointerException();
        
        if (N == q.length) resize(q.length*2);
        
        if (!isEmpty()) last++;
        if (last == q.length) last = 0;
        
        q[last] = item;
        N++;
        
        return;
    }
    public Item removeFirst()                
    // delete and return the item at the front
    {
        Item item; 
        if (isEmpty()) throw new NoSuchElementException();
        
        item = q[first++];
        if (first == q.length) first = 0;
        
        N--;
        if ((N > 0) && (N == q.length/4)) resize(q.length/2);
        
        return item;
    }
    public Item removeLast()                 
    // delete and return the item at the end
    {
        Item item; 
        if (isEmpty()) throw new NoSuchElementException();
        
        item = q[last--];
        if (last == -1) last = q.length-1;
        
        N--;
        if ((N > 0) && (N == q.length/4)) resize(q.length/2);
        
        return item;
    }

    public Iterator<Item> iterator()         
    // return an iterator over items in order from front to end
    {
        return new ArrayIterator();
    }

    private void resize(int n)
    {
        Item[] qq = (Item[]) new Object[n];
        int ii = 0;
        for (Item i : this)
        {
            qq[ii++] = i;
        }
        q = qq;
        last = ii-1;
        first = 0;
        return;
    }
    
    private class ArrayIterator implements Iterator<Item>
    {
        private int i = 0;
        public boolean hasNext() { return i < N; }
        public void remove() {
            /* not supported */
            throw new UnsupportedOperationException();
        }
        public Item next() 
        {
            if (i == N) throw new java.util.NoSuchElementException();
            Item item = q[(first+i) % q.length];
            i++;           
            return item; 
        }
    }

    private String toText()
    {
        String txt = "";
        for (Item i : this)
            txt += i.toString()+',';
        return txt;    
    }
    
    public static void main(String[] args)   // unit testing
    {
        Deque<Integer> d = new Deque<Integer>();
        d.addLast(3);
        d.addLast(4);
        d.addLast(5);
        System.out.println(d.toText());
        d.addFirst(2);
        d.addFirst(1);
        System.out.println(d.toText());

    }
}

