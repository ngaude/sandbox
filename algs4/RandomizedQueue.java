import java.util.Iterator;
import java.util.NoSuchElementException;




public class RandomizedQueue<Item> implements Iterable<Item>  
{
    private Item[] a;         // array of items
    private int N;            // number of elements on queue


    /**
     * Initializes an empty queue.
     */
    public RandomizedQueue() {
        a = (Item[]) new Object[2];
    }

    /**
     * Is this queue empty?
     * @return true if this queue is empty; false otherwise
     */
    public boolean isEmpty() {
        return N == 0;
    }

    /**
     * Returns the number of items in the queue.
     * @return the number of items in the queue
     */
    public int size() {
        return N;
    }


    // resize the underlying array holding the elements
    private void resize(int capacity) {
        assert capacity >= N;
        Item[] temp = (Item[]) new Object[capacity];
        for (int i = 0; i < N; i++) {
            temp[i] = a[i];
        }
        a = temp;
    }

    /**
     * Exchange the i-th and j-th this queue.
     * @param i i-th item index
     * @param j j-th item index
     */
    private void exch(int i, int j)
    {
        assert i < N;
        assert j < N;
        Item swap;
        swap = a[i];
        a[i] = a[j];
        a[j] = swap;
    }

    /**
     * Adds the item to this queue.
     * @param item the item to add
     */
    public void enqueue(Item item) {
        if (N == a.length) resize(2*a.length);    
        // double size of array if necessary
        
        a[N++] = item;                            // add item
        
        exch(N-1, StdRandom.uniform(N));           
        // flush the added item in queue
    }

    /**
     * Removes and returns an item from this queue
     * @return an random item
     * @throws java.util.NoSuchElementException if this queue is empty
     */
    public Item dequeue() {
        if (isEmpty()) throw new NoSuchElementException("queue underflow");
        Item item = a[N-1];
        a[N-1] = null;                              // to avoid loitering
        N--;
        // shrink size of array if necessary
        if (N > 0 && N == a.length/4) resize(a.length/2);
        return item;
    }

    /**
     * Return (but do not delete) a random item.
     * @return a random item added to this queue
     * @throws java.util.NoSuchElementException if this queue is empty
     */
    public Item sample() {
        if (isEmpty()) throw new NoSuchElementException("queue underflow");
        return a[StdRandom.uniform(N)];
    }
    
    /**
     * Returns an iterator to this queue that iterates through 
     * the items in LIFO order.
     * @return an iterator to this queue that iterates through 
     * the items in LIFO order.
     */
    public Iterator<Item> iterator() {
        return new RandomArrayIterator();
    }

    // an iterator, doesn't implement remove() since it's optional
    private class RandomArrayIterator implements Iterator<Item> {
        private int i;
        private int[] o; 

        public RandomArrayIterator() {
            o = new int[N];
            for (i = 0; i < N; i++) {
                o[i] = i;
            }
            StdRandom.shuffle(o);
        }

        public boolean hasNext() {
            return i > 0;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public Item next() {
            if (!hasNext()) throw new NoSuchElementException();
            return a[o[--i]];
        }
    }

    /**
     * Unit tests the <tt>queue</tt> data type.
     */
    public static void main(String[] args) {
        RandomizedQueue<String> s = new RandomizedQueue<String>();
        StdOut.println("(" + s.size() + " left on queue)");
        while (!StdIn.isEmpty()) {
            String item = StdIn.readString();
            if (!item.equals("-")) s.enqueue(item);
            else if (!s.isEmpty()) StdOut.print(s.dequeue() + " ");
        }
        StdOut.println("(" + s.size() + " left on queue)");
    }
 
}