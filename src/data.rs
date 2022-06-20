use serde::{Deserialize, Serialize};

use crate::utils::InstanceData;

pub trait Chunk {
    fn get_block(&self, x: usize, y: usize, z: usize) -> Option<&Block>;
    fn set_block(&mut self, x: usize, y: usize, z: usize, block: Option<Block>);
    // the bits of the u8 tell whether there is air on that side
    // the bits are the sides in order: top, bottom, ... TODO
    fn instance_data_iter<'a>(&'a self) -> Box<dyn Iterator<Item = (u8, InstanceData)> + 'a>;
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

    fn instance_data_iter<'a>(&'a self) -> Box<dyn Iterator<Item = (u8, InstanceData)> + 'a> {
        Box::new(self.data.iter().enumerate().flat_map(move |(x, b)| {
            b.iter().enumerate().flat_map(move |(y, d)| {
                d.iter()
                    .enumerate()
                    .filter_map(|(a, b)| b.as_ref().map(|v| (a, v)))
                    .map(move |(z, f)| {
                        let right_air =
                            x + 1 == self.data.len() || self.data[x + 1][y][z].is_none();
                        let left_air = x == 0 || self.data[x - 1][y][z].is_none();
                        let down_air = y + 1 == self.data.len() || self.data[x][y + 1][z].is_none();
                        let up_air = y == 0 || self.data[x][y - 1][z].is_none();
                        let back_air =
                            z + 1 == self.data[x][y].len() || self.data[x][y][z + 1].is_none();
                        let front_air = z == 0 || self.data[x][y][z - 1].is_none();
                        let bitcode = [right_air, left_air, down_air, up_air, back_air, front_air]
                            .iter()
                            .fold(0, |result, &bit| (result << 1) ^ (bit as u8));
                        return (
                            bitcode,
                            InstanceData {
                                block_type: match f {
                                    Block::Dirt => 0,
                                    Block::Stone => 1,
                                    _ => 1,
                                },
                                position_offset: [
                                    x as f32 * 20.0,
                                    y as f32 * 20.0,
                                    z as f32 * 20.0,
                                ],
                            },
                        );
                    })
            })
        }))
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
