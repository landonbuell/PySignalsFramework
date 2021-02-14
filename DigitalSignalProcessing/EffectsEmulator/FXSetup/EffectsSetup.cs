using System;
using System.Collections.Generic;
using System.Text;

using EffectsEmulator.FXModules;
using EffectsEmulator.FXChain;

namespace EffectsEmulator.FXSetup;
{
    public class EffectsSetup
    {
        // Class for EffectsChain Instance

        #region Members

        public string Name { get; protected set; }
        public string Type { get; protected set; }

        protected EffectsChain

      

        protected int _nModules;

        #endregion

        #region Constructor

        public EffectsSetup(string name)
        {
            // Constructor for Base EffectsChain
            Name = name;
            Type = "EffectsSetup";
        }

        public 

        #endregion

    }
}
