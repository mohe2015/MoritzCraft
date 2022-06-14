use crate::data::{Block, Chunk};

pub trait WorldGeneration {
    fn generate_chunk<C: Chunk>(chunk: &mut C, chunk_x: usize, chunk_y: usize);
}

pub struct RandomWorldGeneration {}

impl WorldGeneration for RandomWorldGeneration {
    fn generate_chunk<C: Chunk>(chunk: &mut C, chunk_x: usize, chunk_y: usize) {
        for x in 0..100 {
            for y in 0..1 {
                for z in 0..100 {
                    chunk.set_block(chunk_x + x, chunk_y + y, z, Some(Block::Dirt))
                }
            }
        }
    }
}

pub struct SuperflatWorldGeneration {}

pub struct PerlinNoiseWorldGeneration {}
