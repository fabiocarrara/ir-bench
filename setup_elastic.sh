#!/bin/bash

mkdir -p esdata

# CONFIG TO ADD
# indices.query.bool.max_clause_count: 5000
# indices.memory.index_buffer_size: 50%
# discovery.type: single-node

docker run --rm -it \
    -p 9200:9200 \
    -p 9300:9300 \
    -v $PWD/esdata:/usr/share/elasticsearch/data \
    -v $PWD/elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml \
    -v $PWD/jvm.options:/usr/share/elasticsearch/config/jvm.options \
    docker.elastic.co/elasticsearch/elasticsearch:7.1.1
    #-e "discovery.type=single-node" \
    #-e "indices.memory.index_buffer_size=50%" \
    #-e "ES_JAVA_OPTS=-Xms8g -Xmx8g" \


