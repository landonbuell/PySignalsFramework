using System;
using System.Collections.Generic;
using System.Text;

using EffectsEmulator.Modules;

namespace EffectsEmulator.FXChain
{
    internal class LinearModuleChain  : BaseModuleChain
    {
        #region members

        private readonly ModuleAbstract _headNode = new ModulePointer("HeadModule");
        private readonly ModuleAbstract _tailNode = new ModulePointer("TailModule");

        #endregion

        #region Constructors

        public LinearModuleChain() :base()
        {
            // Constructor for Empty Effects Chain
            _headNode.Next = _tailNode;
            _tailNode.Prev = _headNode;
        }

        public LinearModuleChain(ModuleAbstract existingModule)
        {
            // Constructor for Single Layer Effects Chain
            _headNode.Next = existingModule;
            existingModule.Prev = _headNode;

            ModuleAbstract endModule = ForwardTraverse(existingModule);
            _tailNode.Prev = endModule;
            endModule.Next = _tailNode;
        }

        public LinearModuleChain(List<ModuleAbstract> existingModules)
        {
            // Constructor Chain with List of Modules
            _headNode.Next = _tailNode;
            _tailNode.Prev = _headNode;
            // Add each node in order to tail
            foreach (ModuleAbstract module in existingModules)
                AppendNode(module);
        }

        public LinearModuleChain(LinearModuleChain existingChain)
        {
            // Constructor for Chain with Existing Chain
            _headNode.Next = existingChain.HeadModule.Next;
            _headNode.Next.Prev = _headNode;

            _tailNode.Prev = existingChain.TailModule.Prev;
            _tailNode.Prev.Next = _tailNode;
        }

        #endregion

        #region LinearChainProperties

        internal ModuleAbstract HeadModule
        {
            // Get Head Module for this Chain
            get { return _headNode; }
        }

        internal ModuleAbstract TailModule
        {
            // Get Tail Module for this Chain
            get { return _tailNode; }
        }

        internal List<ModuleAbstract> ChainList
        {
            // Get LinearEffects
            get
            {
                List<ModuleAbstract> moduleList = new List<ModuleAbstract>();
                ModuleAbstract currentModule = _headNode.Next;
                while (currentModule != _tailNode)
                {
                    moduleList.Add(currentModule);
                    currentModule = currentModule.Next;
                }
                return moduleList;
            }
        }

        public int ChainLength
        {
            // Get The Length of the Module Chain
            get { return ChainList.Count; }
        }

        #endregion

        #region Methods

        internal void AppendNode(ModuleAbstract newModule)
        {
            // Add New Module to end of this chain
            ModuleAbstract oldTail = _tailNode.Prev;
            oldTail.Next = newModule;
            _tailNode.Prev = newModule;
            newModule.Prev = oldTail;
            newModule.Next = _tailNode;
        }

        internal void PrependNode(ModuleAbstract newModule)
        {
            // Add New Module to the start of this chain
            ModuleAbstract oldHead = _headNode.Next;
            oldHead.Prev = newModule;
            _headNode.Next = newModule;
            newModule.Next = oldHead;
            newModule.Prev = _headNode;
        }
        #endregion
    }
}
