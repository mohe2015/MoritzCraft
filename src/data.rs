use serde::{Deserialize, Serialize};

use crate::utils::InstanceData;

pub trait Chunk {
    fn get_block(&self, x: usize, y: usize, z: usize) -> Option<&Block>;
    fn set_block(&mut self, x: usize, y: usize, z: usize, block: Option<Block>);
    fn instance_data_iter<'a>(&'a self) -> Box<dyn Iterator<Item = InstanceData> + 'a>;
}

const CHUNK_SIZE: usize = 16;

#[derive(Serialize, Deserialize, Debug)]
pub struct ChestContents {}

#[derive(Serialize, Deserialize, Debug)]
pub enum Block {
    Dirt,
    Stone,
    Chest(Box<ChestContents>),
}

#[derive(Default, Serialize, Deserialize, Debug)]
pub struct DenseChunk {
    data: [[[Option<Block>; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
}

impl Chunk for DenseChunk {
    fn get_block(&self, x: usize, y: usize, z: usize) -> Option<&Block> {
        return self.data[x][y][z].as_ref();
    }

    fn set_block(&mut self, x: usize, y: usize, z: usize, block: Option<Block>) {
        self.data[x][y][z] = block;
    }

    fn instance_data_iter<'a>(&'a self) -> Box<dyn Iterator<Item = InstanceData> + 'a> {
        return Box::new(self.data.iter().flatten().flatten().map(|e| InstanceData {
            block_type: 1,
            position_offset: [0.0, 0.0, 0.0],
        }));
    }
}

// what we need:
// random access to blocks
// iterating over blocks in order
// storing arbitrary data with special blocks
// maybe multiple blocks at one location (water + half-block) or two half-blocks

// so maybe array of vecs of blocks (so multiple blocks at one location possible)
// or every block that allows another block contains an extension attribute

// storing on disk: the extended attributes at the end in a special area? or maybe just inline because we always read + write whole chunks?
