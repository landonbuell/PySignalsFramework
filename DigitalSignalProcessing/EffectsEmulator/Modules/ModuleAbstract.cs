using System;
using System.Collections.Generic;
using System.Text;

namespace EffectsEmulator.Modules
{
    public abstract class ModuleAbstract
    {
        // Abstract Parent class for all FX Modules

        protected ModuleAbstract _moduleNext;
        protected ModuleAbstract _modulePrev;

        protected int _nSamples;
        protected int _sampleRate;

        public string Name { get; protected set; }
        public string Type { get; protected set; }
        public string Family { get; protected set; }

        public int ChainIndex { get; set; }
        public bool Initialized { get; protected set; }

        #region Constructors

        public ModuleAbstract(string name,int sampleRate = 44100)
        {
            // Constructor for EffectsModule
            Name = name;
            Type = "AbstractEffectsModule";
            Family = "No Family";
        }

        #endregion

        internal ModuleAbstract Next
        {
            // Get or set the Next Module in the Graph
            get { return _moduleNext; }
            set { _moduleNext = value; }
        }

        internal ModuleAbstract Prev
        {
            // Get or Set the Prev Module in the graph
            get { return _modulePrev; }
            set { _modulePrev = value; }
        }

        protected virtual void InitializeModule ()
        {
            // Initialize This Module
            Initialized = true;
        }

        internal virtual float[] Call(float[] X)
        {
            // Call this Effects Module
            return X;
        }


    }
}
