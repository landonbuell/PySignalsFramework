using System;
using System.Collections.Generic;
using System.Text;

namespace EffectsEmulator.Modules
{
    internal class ModulePointer : ModuleAbstract
    {
        // Pointer Module Only Points to another Module
        // Used as Head/Tail Nodes in FX Chain Graph

        internal ModulePointer (string name) : base(name)
        {
            // Constructor for PointerModule
            Type = "PointerModule";
            Next = null;
            Prev = null;
        }

    }
}
