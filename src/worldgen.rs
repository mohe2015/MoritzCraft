use crate::data::{Chunk, Block};



pub trait WorldGeneration {
    fn generate_chunk<C: Chunk>(chunk: C, x: usize, y: usize);
}

pub struct RandomWorldGeneration {

}

impl WorldGeneration for RandomWorldGeneration {
    fn generate_chunk<C: Chunk>(chunk: C, x: usize, y: usize) {
        for x in 0..100 {
            for y in 0..1 {
                for z in 0..100 {
                    chunk.set_block(x, y , z,
                    Block::Dirt)
                }
            }
            }
    }
}


pub struct SuperflatWorldGeneration {

}

pub struct PerlinNoiseWorldGeneration {

}