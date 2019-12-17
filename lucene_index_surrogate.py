import lucene

import time
import numpy as np

from java.nio.file import Paths
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField, FieldType
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader
from org.apache.lucene.search import IndexSearcher, BooleanQuery
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.queryparser.classic import QueryParser


def surrogate_text(feature, k=196):
    '''
    Creates the surrogate text representation of a single feature.
    Args:
        - feature: a single feature with shape (N,)
        - k: length of the truncated permutation; use -1 for the full lenght
    '''
    feature = feature.reshape(-1)
    k = feature.shape[0] if k < 0 else min(k, feature.shape[0])

    # find inverse permutation
    inv_perm = np.argsort(np.argsort(feature)[::-1]) + 1
    # truncate and complement the permutation
    # trunc_inv_perm = (k+1) - np.minimum(inv_perm, (k+1)), or better:
    trunc_inv_perm = np.maximum((k+1) - inv_perm, 0)
    # now each dimension indicates how many times you have to repeat its term in
    # the surrogate text representation

    surrogate = []
    for term, freq in enumerate(trunc_inv_perm):
        if freq:
            surrogate.append('{}^{}'.format(str(term), freq))
        
    return ' '.join(surrogate)


class LuceneIndex (object):

    def __init__(self, lucene_vm, index_dir):
        lucene_vm.attachCurrentThread()
        BooleanQuery.setMaxClauseCount(2**16) # to avoid 'too many boolean clauses'
        
        self.index_dir = index_dir
                
        #self.fields = dict()
        #self.fields['doc_id'] = FieldType(StringField)
        #self.fields['doc_id'].setStored(True)
        #self.fields['doc_id'].setIndexOptions(IndexOptions.DOCS)
        
        #self.fields['content'] = FieldType(TextField)
        # self.fields['content'].setStored(True)
        #self.fields['content'].setIndexOptions(IndexOptions.DOCS_AND_FREQS)
        # self.fields['content'].setStoreTermVectors(True)
        
        self.directory = SimpleFSDirectory(Paths.get(self.index_dir))
        self.analyzer = WhitespaceAnalyzer()
        self.parser = QueryParser('content', self.analyzer)
    	# self.parser.setDefaultOperator(QueryParser.Operator.AND)
    	
        self.writer = None
        self.reader = None
        self.searcher = None
    
    def _init_writer(self):
        # Needed?
        # self.analyzer = LimitTokenCountAnalyzer(self.analyzer, 10000)
        config = IndexWriterConfig(self.analyzer)
        self.writer = IndexWriter(self.directory, config)
    
    def _init_searcher(self):
        self.reader = DirectoryReader.open(self.directory)
        self.searcher = IndexSearcher(self.reader)
        
    def add(self, doc_id, surrogate):
        if self.writer is None:
            self._init_writer()
    
        # surrogate = surrogate_text(feature)
        doc = Document()
        #doc.add(Field('doc_id', doc_id, self.fields['doc_id']))
        #doc.add(Field('content', surrogate, self.fields['content']))
        
        doc.add(Field('doc_id', doc_id, StringField.TYPE_STORED))
        doc.add(Field('content', surrogate, TextField.TYPE_NOT_STORED))
        
        self.writer.addDocument(doc)
        # self.writer.commit()

    def query(self, surrogate, limit=1000):
        if self.searcher is None:
            self._init_searcher()
        
        # surrogate = surrogate_text(feature)
        query = self.parser.parse(surrogate)
        
        # with open('/tmp/query.txt', 'wb') as f:
        #     f.write(surrogate)
            
        # s = time.time()
        scoreDocs = self.searcher.search(query, limit).scoreDocs
        # print('Search Time: {}s'.format(time.time() - s))
        return ((self.searcher.doc(result.doc)['doc_id'], result.score) for result in scoreDocs)
    
    def count(self):
        if self.searcher is None:
            try:
                self._init_searcher()
            except:
                return -1
            
        return self.reader.numDocs()
    
    def close(self):
        if self.writer is not None:
            self.writer.commit()
            self.writer.close()
            self.writer = None
        
        if self.searcher is not None:
            self.searcher = None
            self.reader.close()
            self.reader = None

    '''
    Magic methods to manage this object in a 'with' context.
    This assures that close() is called.
    '''
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == '__main__':
    lucene_vm = lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    with LuceneIndex(lucene_vm, 'debug_index') as idx:
        a = np.arange(5)
        b = np.random.rand(5)
        c = a * 10
        
        #idx.add("1", a)
        #idx.add("2", b)
        
        idx.query(c)
        
        


