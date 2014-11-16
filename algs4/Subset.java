public class Subset {
   public static void main(String[] args) {
       assert args.length < 0;
       int k = Integer.parseInt(args[0]); 
       assert k >= 0;
       
       RandomizedQueue<String> s = new RandomizedQueue<String>();
       while (!StdIn.isEmpty()) {
           String item = StdIn.readString();
           s.enqueue(item);
       }
       
       assert k <= s.size(); 
       while (k-- > 0) {
           String item = s.dequeue();
           StdOut.print(item+'\n');
       }
   }
}

