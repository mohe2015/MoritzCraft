use serde::{Serialize, Deserialize};

pub trait Chunk {
    fn get_block(&self, x: usize, y: usize, z: usize) -> Option<&Block>;
    fn set_block(&self, x: usize, y: usize, z: usize, block: Option<Block>);
}

const CHUNK_SIZE: usize = 32;

#[derive(Serialize, Deserialize, Debug)]
pub struct ChestContents {

}

#[derive(Serialize, Deserialize, Debug)]
pub enum Block {
    Dirt,
    Stone,
    Chest(Box<ChestContents>),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DenseChunk {
    data: [[[Option<Block>; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]
}

impl Chunk for DenseChunk {
    fn get_block(&self, x: usize, y: usize, z: usize) -> Option<&Block> {
        return self.data[x][y][z].as_ref();
    }

    fn set_block(&self, x: usize, y: usize, z: usize, block: Option<Block>) {
        self.data[x][y][z] = block;
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
