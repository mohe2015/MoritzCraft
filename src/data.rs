const CHUNK_SIZE: usize = 32;

struct ChestContents {

}

enum Block {
    Dirt,
    Stone,
    Chest(Box<ChestContents>),
}

struct Chunk {
    data: [[[Block; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]
}

// what we need:
// random access to blocks
// iterating over blocks in order
// storing arbitrary data with special blocks
// maybe multiple blocks at one location (water + half-block) or two half-blocks

// so maybe array of vecs of blocks (so multiple blocks at one location possible)
// or every block that allows another block contains an extension attribute

// storing on disk: the extended attributes at the end in a special area? or maybe just inline because we always read + write whole chunks?
