using System;
using System.Collections.Generic;
using System.Text;

namespace EffectsEmulator.Modules
{
    public abstract class ModuleNBandEQ : ModuleAbstract
    {
        // N - Band Equilizer Class Module
        public int NBands { get; protected set; }

        public ModuleNBandEQ (string name, int nBands, int sampleRate = 44100 ) : base(name, sampleRate)
        {
            // Constructor for NBand
            NBands = nBands;
        }

    }
}
