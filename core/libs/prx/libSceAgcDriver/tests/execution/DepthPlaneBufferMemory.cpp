#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "DepthFastClearHarness.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>

namespace {

using DepthFastClearHarness::Compile;
using DepthFastClearHarness::Covered;
using DepthFastClearHarness::Draw;
using DepthFastClearHarness::Height;
using DepthFastClearHarness::HtileBytes;
using DepthFastClearHarness::Require;
using DepthFastClearHarness::Surface;
using DepthFastClearHarness::Width;

constexpr std::size_t Block = 65536;
alignas(256) std::array<std::uint32_t, HtileBytes / 4u> Htile{};

void* AllocateWatched(std::size_t bytes) {
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, Block);
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* raw = mmap(nullptr, bytes + Block, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(raw != MAP_FAILED, "cannot map the depth plane block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + Block - 1) & ~static_cast<std::uintptr_t>(Block - 1);
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + Block) munmap(reinterpret_cast<void*>(aligned + bytes), begin + Block - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    return block;
}

std::size_t Holding(std::span<const float> plane, float depth) {
    return static_cast<std::size_t>(std::count(plane.begin(), plane.end(), depth));
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto planeBytes = static_cast<std::size_t>(AgcDriver::Graphics::DepthSliceBytes({Width, Height}, 4u));
        Require(planeBytes <= Block, "the depth plane does not fit one block");
        auto* block = static_cast<float*>(AllocateWatched(Block));
        const std::span<float> plane(block, planeBytes / sizeof(float));
        const auto address = reinterpret_cast<std::uintptr_t>(block);
        const auto shaders = Compile(*device);
        const auto reader = Compile(*device, block);
        const Surface depth{address, 0, 0, false, VK_FORMAT_D32_SFLOAT, 0, AgcDriver::Graphics::XorSwizzleMode(AgcDriver::Graphics::TextureTileMode::kZ64KBX)};

        Require(Draw(*device, shaders, depth, {.depthWrite = true, .depthCompare = VK_COMPARE_OP_ALWAYS}) == Covered, "the draw writing depth 0.5 did not cover the target");
        Draw(*device, reader, depth, {.depthTest = false});
        Require(Holding(plane, 0.5f) == static_cast<std::size_t>(Width) * Height, "a vertex fetch from the depth plane did not see the depth 0.5 a draw wrote into it");

        if (!AgcDriver::GuestMemory::WriteWatched() || !AgcDriver::GuestMemory::Watched(address, Block)) {
            std::puts("guest memory is not write-watched here: a buffer write into the depth plane is not exercised");
            std::puts("depth plane buffer memory tests passed");
            return 0;
        }
        std::fill(plane.begin(), plane.end(), 0.75f);
        AgcDriver::GuestMemory::BumpCollectEpoch();
        Require(Draw(*device, shaders, depth, {}) == Covered, "a draw at depth 0.5 failed LESS against the 0.75 written into the depth plane's memory");
        Require(Draw(*device, shaders, depth, {.depthWrite = true}) == Covered, "a draw at depth 0.5 writing depth failed LESS against 0.75");
        Draw(*device, reader, depth, {.depthTest = false});
        Require(Holding(plane, 0.5f) == static_cast<std::size_t>(Width) * Height, "a vertex fetch from the depth plane did not see the depth 0.5 written after the plane was reloaded");

        auto* fresh = static_cast<float*>(AllocateWatched(Block));
        const std::span<float> freshPlane(fresh, planeBytes / sizeof(float));
        const auto freshReader = Compile(*device, fresh);
        const Surface unstored{reinterpret_cast<std::uintptr_t>(fresh), 0, 0, false, VK_FORMAT_D32_SFLOAT, 0, depth.swizzle};
        Require(Draw(*device, shaders, unstored, {.depthWrite = true, .depthCompare = VK_COMPARE_OP_ALWAYS}) == Covered, "the draw writing depth 0.5 into a new plane did not cover the target");
        std::fill(freshPlane.begin(), freshPlane.end(), 0.75f);
        AgcDriver::GuestMemory::BumpCollectEpoch();
        Draw(*device, freshReader, depth, {.depthTest = false});
        Require(Holding(freshPlane, 0.75f) == freshPlane.size(), "the depth 0.5 a draw wrote before a guest write into the plane was stored over that write");
        Require(Draw(*device, shaders, unstored, {}) == Covered, "a draw at depth 0.5 failed LESS against the 0.75 written over depth the surface had not stored");
        std::fill(freshPlane.begin(), freshPlane.end(), 0.25f);
        AgcDriver::GuestMemory::BumpCollectEpoch();
        Require(Draw(*device, shaders, unstored, {}) == 0u, "a draw at depth 0.5 passed LESS against the 0.25 written into the plane before it");
        Require(Draw(*device, shaders, unstored, {.depthWrite = true, .depthCompare = VK_COMPARE_OP_ALWAYS}) == Covered, "the draw writing depth 0.5 over the reloaded plane did not cover the target");
        Draw(*device, freshReader, depth, {.depthTest = false});
        Require(Holding(freshPlane, 0.5f) == static_cast<std::size_t>(Width) * Height, "the depth 0.5 a draw wrote after a guest write into the plane did not reach memory");

        Htile.fill(0xffffffffu);
        auto* cleared = static_cast<float*>(AllocateWatched(Block));
        const std::span<float> clearedPlane(cleared, planeBytes / sizeof(float));
        const auto clearedReader = Compile(*device, cleared);
        const Surface compressed{reinterpret_cast<std::uintptr_t>(cleared), 0, reinterpret_cast<std::uintptr_t>(Htile.data()), false, VK_FORMAT_D32_SFLOAT, 0, depth.swizzle};
        Require(Draw(*device, shaders, compressed, {.depthWrite = true, .depthCompare = VK_COMPARE_OP_ALWAYS}) == Covered, "the draw writing depth 0.5 into a plane with HTILE did not cover the target");
        Draw(*device, clearedReader, depth, {.depthTest = false});
        Require(Holding(clearedPlane, 0.5f) == static_cast<std::size_t>(Width) * Height, "a vertex fetch from a plane with HTILE did not see the depth 0.5 a draw wrote into it");
        std::fill(clearedPlane.begin(), clearedPlane.end(), 0.25f);
        AgcDriver::GuestMemory::BumpCollectEpoch();
        AgcDriver::Graphics::NoteDepthMetadataFill(compressed.htile, HtileBytes, 0u);
        Require(Draw(*device, shaders, compressed, {}) == Covered, "a guest write into the depth plane won over a later HTILE fast clear of it");
        std::puts("depth plane buffer memory tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
