use rand::Rng;

use crate::data::{Block, Chunk};

pub trait WorldGeneration {
    fn generate_chunk<C: Chunk>(&self, chunk: &mut Box<C>, chunk_x: usize, chunk_y: usize);
}

pub struct RandomWorldGeneration {}

impl WorldGeneration for RandomWorldGeneration {
    fn generate_chunk<C: Chunk>(&self, chunk: &mut Box<C>, chunk_x: usize, chunk_y: usize) {
        let mut rng = rand::thread_rng();
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    chunk.set_block(
                        chunk_x + x,
                        chunk_y + y,
                        z,
                        match rng.gen_range(0..5) {
                            0 => Some(Block::Dirt),
                            1 => Some(Block::Stone),
                            _ => None,
                        },
                    )
                }
            }
        }
    }
}

pub struct SuperflatWorldGeneration {}

pub struct PerlinNoiseWorldGeneration {}
