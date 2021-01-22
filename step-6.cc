


/*
 ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000 */


// @sect3{Include files}

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h> //////////////////////////////////////////////////////////////////////////////// table

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/fe_field_function.h>

#include <fstream>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>




using namespace dealii;


// @sect3{The (<code>Step6) class template}

// We have added two new functions to the main class: (<code> set_all_subdomain_objects)
// and (<code> get_fe_function). Since we are creating new objects for each subdomain
// including triangulation, system_rhs, solution, etc., we have to explicitly make each
// aware of the other, which is exactly what is achieved by (<code> set_all_subdomain_objects).
// A new object that helps with this is the vector where these subdomain objects are stored,
// (<code> subdomain_objects). Meanwhile, (<code> get_fe_function) provides a way to retrieve
// the appropriate function to apply as a boundary condition for multiplicative Schwarz.
// Helpful to this function is the vector (<code> solutionfunction_vector), which is
// populated at the end of (<code> solve). More details are provided there.


template <int dim>
class Step6
{
public:
    Step6(const unsigned int s);

    void run(const unsigned int cycle, const std::string method, TableHandler results_table);

    void set_all_subdomain_objects (const std::vector<std::shared_ptr<Step6<dim>>> &objects)
    { subdomain_objects = objects; }

    // std::vector<Vector<double>> solution_vector;
    std::vector<std::unique_ptr<Functions::FEFieldFunction<dim>>> solutionfunction_vector;



private:
    void setup_system();
    void assemble_system(const std::string method);
    Functions::FEFieldFunction<dim> & get_fe_function(const unsigned int boundary_id);
    void solve(TableHandler results_table);
    void refine_grid(TableHandler results_table);

    void output_results(const unsigned int cycle) const;

    const unsigned int s;
    std::vector<std::shared_ptr<Step6<dim>>> subdomain_objects;

    Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;
    SparseMatrix<double> system_matrix;
    SparsityPattern      sparsity_pattern;

    Vector<double> solution;
    Vector<double> system_rhs;


};

// @sect3{The (<code>MyOverlappingBoundaryValues) class template}

// We need to provide interpolate_boundary_values() in Step6::assemble_system a way to
// compute the value to impose as a boundary condition for additive Schwarz. Additive
// Schwarz does not just take one subdomain's solution as the boundary condition on an
// egde as in multiplicative Schwarz; it uses a linear combination of multiple solutions.
// We cannot provide this as a function because the solutions from each subdomain are
// defined on separate triangulations with their own dof_handlers Instead, we can
// provide a way for the boundary condition to be directly computed from these separate
// solutions when given any point in the overlapping region of their domain. In
// assemble_system(), interpolate_boundary_values() provides the Point<dim> to the
// MyOverlappingBoundaryValues object to evaluate the value of the boundary condition
// for that particular point.

// My implementation of Additive Schwarz is broken up in the following way:
//      - the MyOverlappingBoundaryValues class immediately below
//      - the get_overlapping_solution_functions function is directly below the
//          get_fe_function function used to get the solutions to impose as
//          boundary conditions.
//      - the usage of get_overlapping_solution_functions() is in assemble_system()
//          to impose boundary conditions
//      - the initialization of MyOverlappingBoundaryValues objects is done in
//          assemble_system(), immediately before
//          using objects of this type to impose BC for additive Schwarz
// The debugging process is still on-going.


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// begin Part I of additive Schwarz


// MyOverlappingBoundaryValues acts as a blackbox taking in information and outputting the global solution at any given
// point in our domain. More specifically, we must explicitly provide s, boundary_id, and subdomain_objects to create
// a MyOverlappingBoundaryValues object, but do not explicitly provide the points at which we want to compute the
// global solution. These points are provided by interpolate_boundary_values(), which uses the specified boundary_id
// and returns points lying on the edge corresponding to this boundary_id.

template <int dim>
class MyOverlappingBoundaryValues : public Function<dim>
{
public:
    MyOverlappingBoundaryValues (unsigned int s, unsigned int boundary_id,
            std::vector<std::shared_ptr<Step6<dim>>> subdomain_objects);          //to create a MyOverlappingBoundaryValues object, we have to provide s, boundary_id, and subdomain_objects
    std::vector<Functions::FEFieldFunction<dim>> overlapping_solution_functions;  //how we actually define this vector is provided in the MyOverlappingBoundaryValues constructor (overlapping_solution_functions = get_overlapping_solution_functions(boundary_id))
    std::vector<std::shared_ptr<Step6<dim>>> subdomain_objects;
    const unsigned int s;

    virtual double
    value (const Point<dim> &p,                                                  //again, these points are provided by interpolate_boundary_values()
           const unsigned int component = 0) const
    {
        double tau = 0.1;
        // $\tau$ is the variable (determined by the user) used to determine the
        // weight given to the subdomains' new solutions and the remaining weight
        // goes to the equivalent of the old global solution present on the current
        // subdomain, s.

        //Note that, comparing the formula to the code:
        //      R = relevant_subdomains
        //      |R| = relevant_subdomains.size() = overlapping_solution_functions.size() (we do not have access to relevant_subdomains, but DO have access to overlapping_solution_functions)
        //      $\tilde{u}_s = subdomain s's previous solutionfunction evaluated along the particular edge of interest (edge of interest defined by points 'p' given by interpolate_boundary_values())
        //                                  (previous solutionfunction meaning the last entry of solutionfunction_vector if it only has one entry OR
        //                                                                     the second to last entry if solutionfunction_vector has two or more entries)
        //                   = subdomain_objects[s]->solutionfunction_vector.back())->value(p)                                                         if subdomain_objects[s]->solutionfunction_vector.size() == 1   OR
        //                     subdomain_objects[s]->solutionfunction_vector[subdomain_objects[s]->solutionfunction_vector.size() - 2])->value(p)      if subdomain_objects[s]->solutionfunction_vector.size() >= 2

        double solution_on_shared_edge = 0;

        //First portion of my additive Schwarz algorithm for subdomains:
        if (subdomain_objects[s]->solutionfunction_vector.size() == 1) {
            solution_on_shared_edge += (1 - tau * (overlapping_solution_functions.size())+1) *
                    (subdomain_objects[s]->solutionfunction_vector.back())->value(p); //accessing last entry

        } else {
            Assert (subdomain_objects[s]->solutionfunction_vector.size() >= 2,
                    ExcInternalError());
            solution_on_shared_edge += (1 - tau * (overlapping_solution_functions.size()+1)) *
                    (subdomain_objects[s]->solutionfunction_vector[
                            subdomain_objects[s]->solutionfunction_vector.size() - 2])  //accessing second to last entry
                            ->value(p);

        }
        //Second portion of my additive Schwarz algorithm for subdomains:
        for (unsigned int i=0; i < overlapping_solution_functions.size(); ++i) {
            solution_on_shared_edge += tau * overlapping_solution_functions[i].value(p);
        }

        return solution_on_shared_edge;
    }

private:
    std::vector<Functions::FEFieldFunction<dim>> get_overlapping_solution_functions(
            unsigned int boundary_id);

};


template <int dim>
MyOverlappingBoundaryValues<dim>::MyOverlappingBoundaryValues(
        const unsigned int s,
        const unsigned int boundary_id,
        std::vector<std::shared_ptr<Step6<dim>>> subdomain_objects)
        :s(s),
        subdomain_objects(subdomain_objects)
{
            overlapping_solution_functions =
                    get_overlapping_solution_functions(boundary_id);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// end Part I of additive Schwarz


// @sect4{Nonconstant coefficients}

template <int dim>
double coefficient(const Point<dim> &p)
{
    if (p.square() < 0.5 * 0.5)
        return 20;
    else
        return 1;
}



// @sect3{The <code>Step6</code> class implementation}


// @sect4{Step6::Step6}

// Here we have a few additions. First is the initialization of the subdomain on which
// we are preparing to solve.

template <int dim>
Step6<dim>::Step6(const unsigned int s)
        : s(s),
        fe(2),
        dof_handler(triangulation)

{

// Then we manually create the subdomains using their defining corner points:

    const std::vector<Point<2>> corner_points = {
            Point<2>(-0.875, -0.125), Point<2>(0.125, 0.875),
            Point<2>(-0.125, -0.125), Point<2>(0.875, 0.875),
            Point<2>(-0.125,-0.875), Point<2>(0.875,0.125),
            Point<2>(-0.875,-0.875), Point<2>(0.125,0.125)};



    GridGenerator::hyper_rectangle(triangulation,
                                   corner_points[2 * s],
                                   corner_points[2 * s + 1]);

    triangulation.refine_global(2);



// Lastly, we set boundary_ids (those not explicitly set to a value are
// zero by default, a fact that we use later):

    //Set the boundary_ids of edges of subdomain0
    // if (subdomain == 0) {
    if (s == 0) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();

                //Right edge:

                //Set boundary_id to 2 along the portion of gamma2 that makes
                // up part of the right boundary edge of subdomain0
                if ((std::fabs(center(0) - (0.125)) < 1e-12) &&
                    (center(dim - 1) >= 0.125))
                    face->set_boundary_id(2);

                //Set boundary_id to 6 along gamma6, the remaining portion of
                // subdomain0's right edge
                if ((std::fabs(center(0) - (0.125)) < 1e-12) &&
                    (center(dim - 1) < 0.125))
                    face->set_boundary_id(6);

                //Bottom edge:

                //Set boundary_id to 4 along the portion of gamma4 that makes
                // up part of the bottom boundary edge of subdomain0
                if ((std::fabs(center(dim - 1) - (-0.125)) < 1e-12) &&
                    (center(0) < -0.125))
                    face->set_boundary_id(4);

                //Set boundary_id to 8 along gamma8, the remaining portion of
                // subdomain0's bottom edge
                if ((std::fabs(center(dim - 1) - (-0.125)) < 1e-12) &&
                    (center(0) >= -0.125))
                    face->set_boundary_id(8);

                //Remaining edges have boundary_ids of 0 by default.

            }

    //Set the boundary_ids of edges of subdomain1
    //} else if (subdomain == 1) {
    } else if (s == 1) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();

                //Left edge:

                //Set boundary_id to 1 along portion of gamma1 that makes
                // up part of the left boundary edge of subdomain1
                if ((std::fabs(center(0) - (-0.125)) < 1e-12) &&
                    (center(dim - 1) >= 0.125))
                    face->set_boundary_id(1);

                //Set boundary_id to 5 along gamma5, the remaining portion of
                // subdomain1's left edge
                if ((std::fabs(center(0) - (-0.125)) < 1e-12) &&
                    (center(dim - 1) < 0.125))
                    face->set_boundary_id(5);

                //Bottom edge:

                //Set boundary_id to 4 along portion of gamma4 that makes
                // up part of the bottom boundary edge of subdomain1
                if ((std::fabs(center(dim - 1) - (-0.125)) < 1e-12) &&
                    (center(0) >= 0.125))
                    face->set_boundary_id(4);

                //Set boundary_id to 8 along gamma8, the remaining portion of
                // subdomain1's bottom edge
                if ((std::fabs(center(dim - 1) - (-0.125)) < 1e-12) &&
                    (center(0) < 0.125))
                    face->set_boundary_id(8);

                //Remaining edges have boundary_ids of 0 by default.

            }

    //Set the boundary_ids of edges of subdomain2
    //} else if (subdomain == 2) {
    } else if (s == 2) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();

                //Left edge:

                //Set boundary_id to 1 along portion of gamma1 that makes
                // up part of the left boundary edge of subdomain2
                if ((std::fabs(center(0) - (-0.125)) < 1e-12) &&
                    (center(dim - 1) <= -0.125))
                    face->set_boundary_id(1);

                //Set boundary_id to 5 along gamma5, the remaining portion of
                // subdomain2's left edge
                if ((std::fabs(center(0) - (-0.125)) < 1e-12) &&
                    (center(dim - 1) > -0.125))
                    face->set_boundary_id(5);

                //Top edge:

                //Set boundary_id to 3 along portion of gamma3 that makes
                // up part of the top boundary edge of subdomain2
                if ((std::fabs(center(dim - 1) - (0.125)) < 1e-12) &&
                    (center(0) >= 0.125))
                    face->set_boundary_id(3);

                //Set boundary_id to 7 along gamma7, the remaining portion of
                // subdomain2's top edge
                if ((std::fabs(center(dim - 1) - (0.125)) < 1e-12) &&
                    (center(0) < 0.125))
                    face->set_boundary_id(7);

                //Remaining edges have boundary_ids of 0 by default.

            }

    //Set the boundary_ids of edges of subdomain3
    //} else if (subdomain == 3) {
    } else if (s == 3) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();

                //Right edge:

                //Set boundary_id to 2 along portion of gamma2 that makes
                // up part of the right boundary edge of subdomain3
                if ((std::fabs(center(0) - (0.125)) < 1e-12) &&
                    (center(dim - 1) <= -0.125))
                    face->set_boundary_id(2);

                //Set boundary_id to 6 along gamma6, the remaining portion of
                // subdomain3's right edge
                if ((std::fabs(center(0) - (0.125)) < 1e-12) &&
                    (center(dim - 1) > -0.125))
                    face->set_boundary_id(6);

                //Top edge:

                //Set boundary_id to 3 along portion of gamma3 that makes
                // up part of the top boundary edge of subdomain3
                if ((std::fabs(center(dim - 1) - (0.125)) < 1e-12) &&
                    (center(0) <= -0.125))
                    face->set_boundary_id(3);

                //Set boundary_id to 7 along gamma7, the remaining portion
                // of subdomain3's top edge
                if ((std::fabs(center(dim - 1) - (0.125)) < 1e-12) &&
                    (center(0) > -0.125))
                    face->set_boundary_id(7);

                //Remaining edges have boundary_ids of 0 by default.

            }

    } else
        Assert (false, ExcInternalError());


    // Setting the boundary_ids as above is clearly a lengthy, tedious, and bug-prone
    // process that makes the program, as it currently stands, combersome to use.
    // This process will eventually be replaced this with an automated method, but
    // I needed a highly controlled system running before playing with this feature.


    setup_system();

    // (<code> setup_system) makes (<code> solution) of the right size, (<code> n_dofs),
    // and distributes dofs. So after (<code> setup_system), we can make a
    // FEFieldFunction (using solution and dof_handler) equivalent to the zero function
    // (because solution is initialized with a value of zero by default) and make this
    // FEFieldfunction the first entry of each solutionfunction_vector:

    std::unique_ptr<Functions::FEFieldFunction<dim>> zero_function =
            std::make_unique<Functions::FEFieldFunction<dim>>(dof_handler, solution);
    solutionfunction_vector.emplace_back (std::move(zero_function));

}


// @sect4{Step6::setup_system}

// This function remains identical to the original.

template <int dim>
void Step6<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    VectorTools::interpolate_boundary_values(
            dof_handler,0,
            Functions::ZeroFunction<dim>(),
            constraints);

    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);


}


// @sect4{Step6::get_fe_function}

// Retrieves the appropriate (<code> FEFieldFunction) function to be imposed as a
// boundary condition in the multiplicative Schwarz algorithm.
// This association between subdomains, which subdomain solution is imposed as a
// boundary condition to which edge, is also done explicitly here, but will be
// replaced with an automated system.


template<int dim>
Functions::FEFieldFunction<dim> &
        Step6<dim>::get_fe_function(unsigned int boundary_id)
{

    // Get other subdomain id
    types::subdomain_id relevant_subdomain;

        if (s == 0) {
            if (boundary_id == 2)
                relevant_subdomain = 1;

            else if (boundary_id == 6)
                relevant_subdomain = 2;

            else if (boundary_id == 8)
                relevant_subdomain = 3;

            else if (boundary_id == 4)
                relevant_subdomain = 3;

            else //boundary_id == 0
            Assert (false, ExcInternalError());


        } else if (s == 1) {
            if (boundary_id == 1)
                relevant_subdomain = 0;

            else if (boundary_id == 5)
                relevant_subdomain = 0;

            else if (boundary_id == 8)
                relevant_subdomain = 3;

            else if (boundary_id == 4)
                relevant_subdomain = 2;

            else //boundary_id == 0
            Assert (false, ExcInternalError());


        } else if (s == 2) {

            if (boundary_id == 3)
                relevant_subdomain = 1;

            else if (boundary_id == 5)
                relevant_subdomain = 0;

            else if (boundary_id == 7)
                relevant_subdomain = 1;

            else if (boundary_id == 1)
                relevant_subdomain = 3;

            else //boundary_id == 0
            Assert (false, ExcInternalError());


        } else if (s == 3) {

            if (boundary_id == 3)
                relevant_subdomain = 0;

            else if (boundary_id == 6)
                relevant_subdomain = 2;

            else if (boundary_id == 7)
                relevant_subdomain = 1;

            else if (boundary_id == 2)
                relevant_subdomain = 2;

            else //boundary_id == 0
            Assert (false, ExcInternalError());


        } else Assert (false, ExcInternalError());


        //std::cout << "              The BC function for subdomain " << s <<
        // " on gamma" << boundary_id << " is from subdomain " <<
        // relevant_subdomain << std::endl;

        //For Multiplicative Schwarz, we impose the most recently computed solution
        // from neighboring subdomains as the BC of the current subdomain, so we
        // retrieve the last entry of the appropriate solutionfunction_vector:

        return *subdomain_objects[relevant_subdomain]->solutionfunction_vector.back();




}

// @sect4{MyOverlappingBoundaryValues<dim>::get_overlapping_solution_functions}

// Retrieves the appropriate (<code> FEFieldFunction) functions to be imposed as a
// boundary conditions in the additive Schwarz algorithm.
// This association between subdomains, which subdomain solution is imposed as a
// boundary condition to which edge, is also done explicitly here, but will be
// replaced with an automated system.

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// begin Part II of additive Schwarz


template<int dim>
std::vector<Functions::FEFieldFunction<dim>>
MyOverlappingBoundaryValues<dim>::get_overlapping_solution_functions(
        unsigned int boundary_id)
{

    // We have to define the set R from my formula for additive Schwarz with
    // subdomains mentioned in the introduction. Here, we implement this
    // set as a vector, (<code> relevant_subdomains).
    std::vector<int> relevant_subdomains;

    if (s == 0) {
        if (boundary_id == 2) {
            relevant_subdomains = {1};

        } else if (boundary_id == 6) {
            relevant_subdomains = {2,1,3};

        } else if (boundary_id == 8) {
            relevant_subdomains = {3,1,2};

        } else if (boundary_id == 4) {
            relevant_subdomains = {3};

        } else //boundary_id == 0
        Assert (false, ExcInternalError());


    } else if (s == 1) {
        if (boundary_id == 1) {
            relevant_subdomains = {0};

        } else if (boundary_id == 5) {
            relevant_subdomains = {0,2,3};

        } else if (boundary_id == 8) {
            relevant_subdomains = {3,0,2};

        } else if (boundary_id == 4) {
            relevant_subdomains = {2};

        } else //boundary_id == 0
        Assert (false, ExcInternalError());


    } else if (s == 2) {

        if (boundary_id == 3)
            relevant_subdomains = {1};

        else if (boundary_id == 5)
            relevant_subdomains = {0,1,3};

        else if (boundary_id == 7)
            relevant_subdomains = {1,0,3};

        else if (boundary_id == 1)
            relevant_subdomains = {3};

        else //boundary_id == 0
        Assert (false, ExcInternalError());


    } else if (s == 3) {

        if (boundary_id == 3)
            relevant_subdomains = {0};

        else if (boundary_id == 6)
            relevant_subdomains = {2,0,1};

        else if (boundary_id == 7)
            relevant_subdomains = {1,0,2};

        else if (boundary_id == 2)
            relevant_subdomains = {2};

        else //boundary_id == 0
        Assert (false, ExcInternalError());


    } else Assert (false, ExcInternalError());


    // Now we have to retrieve $\tilde{u}_{i}^{\{k+1\}}$ for i $\in$ R and
    // $\tilde{u}_{s}^{\{k\}}$. When solving in parallel, these are the last entries
    // of subdomain_objects[relevant_subdomains[i]]->solutionfunction_vector and the
    // second to last entry of subdomain_objects[s]->solutionfunction_vector
    // respectively.
    // When solving on subdomains sequentially instead, we must consider if we have
    // solved on relevant_subdomains[i] during the current cycle. If
    // relevant_subdomains[i] >= s, the solution on this subdomain has not yet been
    // computed in this cycle, so we retrieve the last entry of
    // relevant_subdomains[i]'s solutionfunction_vector.
    // Otherwise, relevant_subdomains[i] < s and the solution has been computed on
    // this subdomain in the current cycle, so we must retrieve the second to last
    // entry of relevant_subdomains[i]'s solutionfunction_vector.
    // We also must consider the fact that the second to last entry of
    // solutionfunction_vector may not exist; namely when the size of
    // solutionfunction_vector is 1. In this case, we need to retrieve its only entry,
    // which happens to be the vector's last entry and can be accesed with .back().
    // Whether solving in parallel or not, $\tilde{u}_{s}^{\{k\}}$ is always the last
    // entry of subdomain_objects[s]->solutionfunction_vector at this point in time.


    for (unsigned int i=0; i < relevant_subdomains.size(); i++) {
        if ((subdomain_objects[relevant_subdomains[i]]->solutionfunction_vector.size())
        == 1) {
            overlapping_solution_functions.push_back(
                    *subdomain_objects[relevant_subdomains[i]]->
                    solutionfunction_vector.back());

        } else {
            //Assert (subdomain_objects[s]->solutionfunction_vector.size() >= 2,
            //        ExcInternalError());
            Assert (subdomain_objects[relevant_subdomains[i]]->solutionfunction_vector.size() >= 2,
                    ExcInternalError());

            if (relevant_subdomains[i] >= s) {
                //In this case, the solution for relevant_subdomain has not been
                // computed in the current cycle, so retrieving the last element
                // of its solutionfunction_vector is retrieving its solution from
                // the previous cycle
                overlapping_solution_functions.push_back(
                        *subdomain_objects[relevant_subdomains[i]]->
                        solutionfunction_vector.back());

            } else { //relevant_subdomains[i] < s
                // Here, relevant_subdomain < s and the solution for relevant_subdomain
                // has been computed in the current cycle, so retrieving the second to
                // last element of its solutionfunction_vector is retrieving its
                // solution from the previous cycle
                Assert (relevant_subdomains[i] < s,
                        ExcInternalError());
                overlapping_solution_functions.push_back(
                        *subdomain_objects[relevant_subdomains[i]]->
                        solutionfunction_vector[subdomain_objects[
                                relevant_subdomains[i]]->solutionfunction_vector.size()-2]);
            }

        }

    }

    return overlapping_solution_functions;

    // Now we have overlapping_solution_functions, a std::vector of Vector<double>'s.
    // So we need to:
    //     (1) Iterate through overlapping_solution_functions
    //     (2) Compute the current element's value at a Point<dim>, which will later
    //          be provided by VectorTools::interpolate_boundary_values() at the time
    //          of imposing boundary conditions (which happens in Step6::assemble_system())
    //     (3) Multiply the result by the appropriate constant
    //     (4) Add all of these results together.
    // All of the above are done in the MyOverlappingBoundaryValues<dim> class constructor.


}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// end Part II additive Schwarz


// @sect4{Step6::assemble_system}

// Next, we actually impose the appropriate solutions as boundary conditions on their
// appropriate edges, which is done with VectorTools::interpolate_boundary_values.
// We specify the function that will be applied as a boundary condition on the nonzero
// boundary_ids using get_fe_function for multiplicative Schwarz and
// get_overlapping_solution_functions for additive Schwarz. but on the boundaries
// with boudary_id of zero by default (the edges on the exterior of the entire domain)
// we impose the homogenous Direchlet boundary condition by using
// Functions::ZeroFunction<dim>() as the appropriate boundary function.


template <int dim>
void Step6<dim>::assemble_system(const std::string method)
{
    system_matrix = 0;
    system_rhs = 0;

    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit(cell);
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            const double current_coefficient =
                    coefficient<dim>(fe_values.quadrature_point(q_index));
            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices())
                    cell_matrix(i, j) +=
                            (current_coefficient *              // a(x_q)
                             fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                             fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                             fe_values.JxW(q_index));           // dx
                cell_rhs(i) += (1.0 *                               // f(x)
                                fe_values.shape_value(i, q_index) * // phi_i(x_q)
                                fe_values.JxW(q_index));            // dx
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
                cell_matrix, cell_rhs, local_dof_indices,
                system_matrix, system_rhs);
    }


    std::map<types::global_dof_index, double> boundary_values;

    //Impose boundary condition on edges with boundary_id of 0
    VectorTools::interpolate_boundary_values(
            dof_handler,
            0,
            Functions::ZeroFunction<dim>(),
            boundary_values);


    //Impose boundary conditions for multiplicative Schwarz

    if (method == "Multiplicative") {


        //Impose boundary conditions on edges of subdomain0 with nonzero
        // boundary_ids
        if (s == 0) {

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    2,
                    get_fe_function(2),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    4,
                    get_fe_function(4),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    6,
                    get_fe_function(6),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    8,
                    get_fe_function(8),
                    boundary_values);


        //Impose boundary conditions on edges of subdomain1 with nonzero
        // boundary_ids
        } else if (s == 1) {

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    1,
                    get_fe_function(1),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    4,
                    get_fe_function(4),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    5,
                    get_fe_function(5),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    8,
                    get_fe_function(8),
                    boundary_values);

        //Impose boundary conditions on edges of subdomain2 with nonzero
        // boundary_ids
        } else if (s == 2) {
            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    1,
                    get_fe_function(1),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    3,
                    get_fe_function(3),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    5,
                    get_fe_function(5),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    7,
                    get_fe_function(7),
                    boundary_values);

            //Impose boundary conditions on edges of subdomain3 with nonzero
            // boundary_ids
        } else if (s == 3) {
            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    2,
                    get_fe_function(2),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    3,
                    get_fe_function(3),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    6,
                    get_fe_function(6),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    7,
                    get_fe_function(7),
                    boundary_values);

        } else Assert (false, ExcInternalError());



    //Impose boundary conditions for additive Schwarz

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// begin Part III additive Schwarz
/*



    } else if (method == "Additive") {

        //Impose boundary conditions on edges of subdomain0 with nonzero
        // boundary_ids
        if (s == 0) {
            MyOverlappingBoundaryValues<dim> overlapping_subdomains_2(
                    s, 2, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     2,
                                                     overlapping_subdomains_2,
                                                     boundary_values);

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_4(
                    s, 4, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     4,
                                                     overlapping_subdomains_4,
                                                     boundary_values);

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_6(
                    s, 6, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     6,
                                                     overlapping_subdomains_6,
                                                     boundary_values);

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_8(
                    s, 8, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     8,
                                                     overlapping_subdomains_8,
                                                     boundary_values);


            //Impose boundary conditions on edges of subdomain1 with nonzero
            // boundary_ids


        } else if (s == 1) {

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_1(
                    s, 1, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     1,
                                                     overlapping_subdomains_1,
                                                     boundary_values);

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_4(
                    s, 4, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     4,
                                                     overlapping_subdomains_4,
                                                     boundary_values);

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_5(
                    s, 5, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     5,
                                                     overlapping_subdomains_5,
                                                     boundary_values);

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_8(
                    s, 8, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     8,
                                                     overlapping_subdomains_8,
                                                     boundary_values);

            //Impose boundary conditions on edges of subdomain2 with nonzero
            // boundary_ids
        } else if (s == 2) {

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_1(
                    s, 1, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     1,
                                                     overlapping_subdomains_1,
                                                     boundary_values);

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_3(
                    s, 3, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     3,
                                                     overlapping_subdomains_3,
                                                     boundary_values);

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_5(
                    s, 5, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     5,
                                                     overlapping_subdomains_5,
                                                     boundary_values);

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_7(
                    s, 7, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     7,
                                                     overlapping_subdomains_7,
                                                     boundary_values);

            //Impose boundary conditions on edges of subdomain3 with nonzero
            // boundary_ids
        } else if (s == 3) {

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_2(
                    s, 2, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     2,
                                                     overlapping_subdomains_2,
                                                     boundary_values);

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_3(
                    s, 3, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     3,
                                                     overlapping_subdomains_3,
                                                     boundary_values);

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_6(
                    s, 6, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     6,
                                                     overlapping_subdomains_6,
                                                     boundary_values);

            MyOverlappingBoundaryValues<dim> overlapping_subdomains_7(
                    s, 7, subdomain_objects);
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     7,
                                                     overlapping_subdomains_7,
                                                     boundary_values);

        } else Assert (false, ExcInternalError());
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// end Part III additive Schwarz

    } else { //Neither Multiplicative Schwarz nor Additive Schwarz were
        // chosen as the solving method
        std::cout <<
        "Error: 'Multiplicative' or 'Additive' must be chosen as the solving method"
        << std::endl;
        Assert (false, ExcInternalError());

    }

    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);


}


// @sect4{Step6::solve}

// The same solver as before is used because this is not the focus of this
// modified program.

template <int dim>
void Step6<dim>::solve(TableHandler results_table)
{
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

// Here we store the solution as function that can be retrieved elsewhere.
// Namely, this will be used by get_fe_function and get_overlapping_solution_functions.
    std::unique_ptr<Functions::FEFieldFunction<dim>> solutionfunction_pointer =
            std::make_unique<Functions::FEFieldFunction<dim>>(dof_handler, solution);

    solutionfunction_vector.emplace_back (std::move(solutionfunction_pointer));

    std::cout << "  max solution value=" << solution.linfty_norm()
              << std::endl;
    results_table.add_value("MaxSolValue", solution.linfty_norm()); //////////////////////////////////////////table

}



// @sect4{Step6::refine_grid}

template <int dim>
void Step6<dim>::refine_grid(TableHandler results_table)
{
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       {},
                                       solution,
                                       estimated_error_per_cell);

    std::cout << " Max error present:" << estimated_error_per_cell.linfty_norm() <<
    std::endl;
    results_table.add_value("MaxError", estimated_error_per_cell.linfty_norm()); ///////////////////////////////////table

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.03);

    triangulation.execute_coarsening_and_refinement();


}



// @sect4{Step6::output_results}

template <int dim>
void Step6<dim>::output_results(const unsigned int cycle) const
{

    {

        GridOut grid_out;

        std::ofstream output("grid-" + std::to_string(s * 100 + cycle) + ".gnuplot");

        GridOutFlags::Gnuplot gnuplot_flags(false, 5);
        grid_out.set_flags(gnuplot_flags);
        MappingQGeneric<dim> mapping(3);
        grid_out.write_gnuplot(triangulation, output, &mapping);

    }

    {

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches();

        std::ofstream output("solution-" + std::to_string(s * 100 + cycle) + ".vtu");

        data_out.write_vtu(output);

    }

}


// @sect4{Step6::run}

// The (<code> run) function remains nearly identical; some of the functions it calls
// merely needed extra arguments.

template <int dim>
void Step6<dim>::run(const unsigned int cycle, const std::string method, TableHandler results_table) {

    std::cout << "Cycle:  " << cycle << std::endl;
    std::cout << "Subdomain:  " << s << std::endl;

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    refine_grid(results_table);

    setup_system();

    std::cout << " After calling refine_grid():" << std::endl;

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;
    results_table.add_value("ActiveCells", triangulation.n_active_cells()); ////////////////////////////////// table

    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;
    results_table.add_value("DoF", dof_handler.n_dofs());  /////////////////////////////////////////////////// table


    //assemble_system(s, method);
    assemble_system(method);


        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;
        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() <<
        std::endl;

    solve(results_table);

    output_results(cycle);

}


int main()
{
    try
    {
        // We now construct each subdomain problem and store them all in a vector
        // to solve iteratively.
        std::vector<std::shared_ptr<Step6<2>>> subdomain_problems;

        subdomain_problems.push_back (std::make_shared<Step6<2>> (0));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (1));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (2));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (3));


        // Tell each of the objects representing one subdomain each about the objects
        // representing all of the other subdomains:
        for (unsigned int s=0; s<subdomain_problems.size(); ++s) {
            subdomain_problems[s] -> set_all_subdomain_objects(subdomain_problems);
        }

        // Choose whether we use multiplicative or additive Schwarz to solve
        std::string method;
        std::cout <<
        "Which Schwarz method would you like to use? (Multiplicative or Additive)" <<
        std::endl;

        std::cin >> method;

        TableHandler results_table; //////////////////////////////////////////////////////////////////////////////////// table
        Timer timer;

        // Now we can actually solve each subdomain problem
        for (unsigned int cycle=1; cycle<10; ++cycle)
            for (unsigned int s=0; s<subdomain_problems.size(); ++s) {
                subdomain_problems[s] -> run(cycle, method, results_table);

                std::cout << "After solving on subdomain " << s <<
                " during cycle " << cycle << ":\n" <<
                "        Elapsed CPU time: " << timer.cpu_time() <<
                " seconds.\n";

                std::cout << "        Elapsed wall time: " << timer.wall_time() <<
                " seconds.\n";


                results_table.add_value("Cycle", cycle);                      ////////////////////////////////////// table
                results_table.add_value("Subdomain", s);                      ////////////////////////////////////// table
                results_table.add_value("CPUtimeTotal", timer.cpu_time());    ////////////////////////////////////// table
                results_table.add_value("WalltimeTotal", timer.wall_time());  ////////////////////////////////////// table

            }

        timer.stop();
        timer.reset();

        std::ofstream out_file("number_table.tex");
        results_table.write_tex(out_file);
        out_file.close();

    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}




