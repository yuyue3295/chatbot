from threading import Thread
from queue import Queue

class ThreadedGenerator(object):
    '''
    是用这个类的对象来生成数据，因为是多线程的缘故，所以效率会比较高，
    每给一个迭代器就会有生成一个队列，一个线程负责将数据放入到队列中去。后面对象用作生成器。
    防止系统卡顿，所有的新数据都在新的线程里面去做。

    '''
    def __init__(self,iterator,sentinel = object(),queue_maxsize=0,
                 daemon = False):
        self._iterator = iterator
        self._sentinel = sentinel
        self._queue = Queue(maxsize=queue_maxsize)
        self._thread = Thread(name=repr(iterator),target=self._run)
        self._thread.daemon = daemon
        self._started = False

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def _run(self):
        '''
        这个是线程需要运行的方法，将迭代器中的数据put都队列中去，最后放入一个对象作为哨兵。
        '''

        '''
        :return:
        '''
        try:
            for value in self._iterator:
                if not self._started:
                    return
                self._queue.put(value)
        except:
            print('xixi')

        finally:
            self._queue.put(self._sentinel)#队列里面最后添加一个对象作为哨兵。


    def close(self):
        '''
        循环将队列里的数据清空
        :return:
        '''
        self._started = False
        try:
            while True:
                self._queue.get(timeout=30)
        except KeyboardInterrupt as e:
            raise  e
        except:
            pass


    def __iter__(self):
        '''
        生成器，并没有全部加入到内存中，需要的时候开始计算下一个需要返回的值
        :return:
        '''
        self._started = True
        self._thread.start()
        for value in iter(self._queue.get,self._sentinel):
            yield value
        #yield，做了一个位置的记录，

        self._thread.join()
        self._started = False

    def __next__(self):
        '''从队列中获取一个值'''
        if not self._started:
            self._started = True
            self._thread.start()
        value = self._queue.get(timeout=30)
        if value == self._sentinel:
            raise StopIteration()
        return value

def test():
    def gene():
        i = 0
        while True:
            yield i
            i = i + 1

    t = gene()
    test = ThreadedGenerator(t)
    for _ in range(20):
        print(next(test))

    test.close()

if __name__ == '__main__':
    test()


